import sys


# if sys.platform == "linux":
import socket
import select

# else:
import serial

import time
import re
from math import log2
import numpy as np

from scientisst.frame import *
from scientisst.state import *
from scientisst.exceptions import *
from scientisst.esp_adc.esp_adc import *
from scientisst.constants import *


class ScientISST:
    """ScientISST Device class

    Attributes:
        address (str): The device serial port address ("/dev/example") or TCP port

        serial_speed (int, optional): The serial port bitrate.
    """

    __serial = None
    __socket = None
    __num_chs = 0
    __api_mode = 1
    __sample_rate = None
    __chs = [None] * 8
    __log = False

    def __init__(
        self,
        address,
        serial_speed=115200,
        log=False,
        api=API_MODE_SCIENTISST,
        connection_tries=5,
        com_mode=COM_MODE_BT,
    ):
        """
        Args:
            address (str): The device serial port address ("/dev/example")
            serial_speed (int, optional): The serial port bitrate in bit/s
            log (bool, optional): If the bytes sent and received should be showed
            api (int): The desired API mode for the device
        """

        if (
            api != API_MODE_SCIENTISST
            and api != API_MODE_JSON
            and api != API_MODE_BITALINO
            and api != API_MODE_SCIENTISST_V2
        ):
            raise InvalidParameterError()

        self.com_mode = com_mode
        self.address = address
        self.serial_speed = serial_speed
        self.__log = log

        self.__serial = None
        self.__socket = None
        self.__num_chs = 0
        self.__api_mode = 1
        self.__sample_rate = None
        self.__chs = [None] * 8
        self.__log = False

        # Setup socket in function of com_mode argument
        self.__setupSocket()

        # try to connect to board
        while True:
            try:
                # Set API mode
                self.__changeAPI(api)
                # get device version string and adc characteristics
                self.version_and_adc_chars()
                break
            except ContactingDeviceError:
                if connection_tries > 0:
                    connection_tries -= 1
                else:
                    raise ContactingDeviceError()

        sys.stdout.write("Connected!\n")

    def version_and_adc_chars(self, print=True):
        """
        Gets the device firmware version string and esp_adc_characteristics

        Returns:
            version (str): Firmware version

        Raises:
            ContactingDeviceError: If there is an error contacting the device.
        """
        if self.__api_mode == API_MODE_BITALINO:
            header = "BITalino"
        else:
            header = ""
        header_len = len(header)

        cmd = b"\x07"
        self.__send(cmd)

        result = self.__recv(1024, waitall_flag=False)

        if result == b"":
            raise ContactingDeviceError()

        index = result.index(b"\x00")
        version = result[header_len: index].decode("utf-8")

        self.__adc1_chars = EspAdcCalChars(result[index + 1:])

        if print:
            sys.stdout.write("ScientISST version: {}\n".format(version))
            sys.stdout.write(
                "ScientISST Board Vref: {}\n".format(self.__adc1_chars.vref)
            )
            sys.stdout.write(
                "ScientISST Board ADC Attenuation Mode: {}\n".format(
                    self.__adc1_chars.atten
                )
            )

        return version

    def start(
        self,
        sample_rate,
        channels,
        reads_per_second=5,
        simulated=False,
    ):
        """
        Starts a signal acquisition from the device

        Args:
            sample_rate (int): Sampling rate in Hz.

                Accepted values are 1, 10, 100 or 1000 Hz.

            channels (list): Set of channels to acquire.

                Accepted channels are 1...6 for inputs A1...A6.

            reads_per_second (int): Number of times to read the data streaming from the device.

                Accepted values are integers greater than 0.


            simulated (bool): If true, start in simulated mode.

                Otherwise start in live mode. Default is to start in live mode.

        Raises:
            DeviceNotIdleError: If the device is already in acquisition mode.
            InvalidParameterError: If no valid API value is chosen or an incorrect array of channels is provided.
        """
        assert int(reads_per_second) > 0

        if self.__num_chs != 0:
            raise DeviceNotIdleError()

        if not channels:  # channels is empty
            chMask = 0xFF  # all 8 analog channels
            self.__num_chs = 8
        else:
            chMask = 0
            for ch in channels:
                if ch <= 0 or ch > 8:
                    raise InvalidParameterError()
                self.__chs[self.__num_chs] = ch  # Fill chs vector

                mask = 1 << (ch - 1)
                if chMask & mask:
                    self.__num_chs = 0
                    raise InvalidParameterError()

                chMask |= mask
                self.__num_chs += 1

        self.__sample_rate = sample_rate

        # Sample rate
        sr = 0b01000011
        sr |= self.__sample_rate << 8
        self.__send(sr, 4)

        # Cleanup existing data in bluetooth socket
        self.__clear()

        if simulated:
            cmd = 0x02
        else:
            cmd = 0x01
        cmd |= chMask << 8

        self.__send(cmd)

        self.__packet_size = self.__getPacketSize()

        self.__bytes_to_read = self.__packet_size * max(
            sample_rate // reads_per_second, 1
        )
        if self.__bytes_to_read > MAX_BUFFER_SIZE:
            self.__bytes_to_read = MAX_BUFFER_SIZE - (
                MAX_BUFFER_SIZE % self.__packet_size
            )

        if self.__bytes_to_read % self.__packet_size:
            self.__num_chs = 0
            sys.stderr.write(
                "Error, bytes_to_read needs to be devisible by packet_size\n"
            )
            raise InvalidParameterError()
        else:
            self.__num_frames = self.__bytes_to_read // self.__packet_size

    def read_test(self, convert=False):
        """
        Reads and decodes data from the device in API mode V1.

        :param num_frames: Number of frames to read
        :param convert: Whether to convert raw data to voltage
        """
        result = bytearray()
        frame_list = []

        num_frames = self.__num_frames

        # Receive the required data for the specified number of frames
        total_bytes = num_frames * self.__packet_size
        while len(result) < total_bytes:
            result += self.__recv(total_bytes - len(result))

        start = 0

        for it in range(num_frames):
            bf = result[start: start + self.__packet_size]
            mid_frame_flag = 0

            # Print raw bytes for the current frame
            print(f"Frame {it}: Raw bytes: {bf}")

            # Check CRC and resynchronize if necessary
            while not self.__checkCRC4(bf, self.__packet_size):
                print(f"Frame {it}: CRC check failed. Raw bytes: {bf}")
                result_tmp = list(self.__recv(1))
                if len(result_tmp) != 1:
                    raise ContactingDeviceError()
                result += result_tmp
                start += 1
                bf = result[start: start + self.__packet_size]
                print(f"Frame {it}: New synchronized bytes: {bf}")

            # Extract sequence and digital I/O states
            if self.__api_mode == API_MODE_SCIENTISST:
                seq = (bf[-1] << 28) | (bf[-2] << 20) | (bf[-3] << 12) | (bf[-4] << 4) | ((bf[-5] & 0xF0) >> 4)
                dio = bf[-5] & 0x0F
                print(f"Frame {it}: Sequence number: {seq}, Digital I/O: {dio}")

            byte_it = 0
            frame_data = {"a": [0] * self.__num_chs, "mv": [0.0] * self.__num_chs}

            for i in range(self.__num_chs):
                index = self.__num_chs - 1 - i
                curr_ch = self.__chs[index]
                print(f"Frame {it}: Channel {i}, Index: {index}, Type: {curr_ch}")

                if curr_ch == AX1 or curr_ch == AX2:  # AX channels
                    frame_data["a"][index] = (
                            int.from_bytes(bf[byte_it: byte_it + 4], byteorder="little") & 0xFFFFFF
                    )
                    byte_it += 3
                    print(f"Frame {it}: AX Channel {i}: Raw value: {frame_data['a'][index]}")

                    if convert:
                        frame_data["mv"][index] = (
                                                          (frame_data["a"][index]) * (3.3 * 2) / (pow(2, 24) - 1)
                                                  ) * 1000
                        frame_data["mv"][index] = round(frame_data["mv"][index], 3)
                        print(f"Frame {it}: AX Channel {i}: Converted value: {frame_data['mv'][index]} mV")

                else:  # AI channels
                    if not mid_frame_flag:
                        frame_data["a"][index] = (
                                int.from_bytes(bf[byte_it: byte_it + 2], byteorder="little") & 0xFFF
                        )
                        byte_it += 1
                        mid_frame_flag = 1
                    else:
                        frame_data["a"][index] = (
                                int.from_bytes(bf[byte_it: byte_it + 2], byteorder="little") >> 4
                        )
                        byte_it += 2
                        mid_frame_flag = 0

                    print(f"Frame {it}: AI Channel {i}: Raw value: {frame_data['a'][index]}")

                    if convert:
                        frame_data["mv"][index] = self.__adc1_chars.esp_adc_cal_raw_to_voltage(frame_data["a"][index])
                        print(f"Frame {it}: AI Channel {i}: Converted value: {frame_data['mv'][index]} mV")

            frame_list.append(frame_data)
            start += self.__packet_size

        return frame_list

    def extract_subchunk_from_frame(self, frames: list, idx_extract=6):
        """This function extracts the samples of a frame that belong to a specified channel.

        Parameters
        ----------
        frames : list
            The frame data chunk.
        idx_extract : int
            The index of the channel within the frame.

        Returns
        -------
         sub_chunk : np.ndarray
            The filtered ACC signal.
            """""
        list_vals = []
        for frame in frames:
            list_vals.append(frame.to_matrix()[idx_extract])

        sub_chunk = np.asarray(list_vals)
        return sub_chunk

    def dac_control(self, dac_value, frames, tick):

        if tick % 5 == 0:

            # we extract channel AX7 (index 6 in the frame) to get the EDA signal
            eda_signal = self.extract_subchunk_from_frame(frames, idx_extract=6)
            mean_eda = np.mean(eda_signal)

            self.scale_factor = 10 ** 3

            dac_condition = 0 < dac_value < 255
            if (mean_eda < 300000) and dac_condition:
                dac_value -= 1
            elif (mean_eda > 7000000) and dac_condition:
                dac_value += 1
            self.dac(dac_value, pwm=True)

        return dac_value

    def read(self, curr_dac_value=None, convert=True, matrix=False):
        """
        # AI1: ACC (1)
        # AI2: DAC [WORKAROUND]
        # AI3: ACC (2)
        # AI6: canal extra (e.g., PPG)
        # AX7: EDA
        # channels = [1,2,3,6,7]

        Reads acquisition frames from the device.

        This method returns when all requested frames are received from the device, or when a timeout occurs.

        Args:
            convert (bool): Convert from raw to mV
            matrix (bool): Return `Frames` in a `np.array` (matrix) form

        Returns:
            frames (list): List of [`Frame`][scientisst.frame.Frame] objects retrieved from the device. If `matrix` is True, the `frames` corresponds to a `np.array` (matrix).

        Raises:
            ContactingDeviceError: If there is an error contacting the device.
            DeviceNotInAcquisitionError: If the device is not in acquisition mode.
            NotSupportedError: If the device API is in BITALINO mode
            UnknownError: If the device stopped sending frames for some unknown reason.
        """

        frames = []

        if self.__num_chs == 0:
            raise DeviceNotInAcquisitionError()

        result = list(self.__recv(self.__bytes_to_read))
        start = 0
        for it in range(self.__num_frames):
            bf = result[start: start + self.__packet_size]
            mid_frame_flag = 0

            #  if CRC check failed, try to resynchronize with the next valid frame
            while not self.__checkCRC4(bf, self.__packet_size):
                sys.stderr.write("Error checking CRC4\n")
                #  checking with one new byte at a time
                result_tmp = list(self.__recv(1))
                if len(result_tmp) != 1:
                    raise ContactingDeviceError()

                result += result_tmp
                start += 1
                bf = result[start: start + self.__packet_size]

            f = Frame(self.__num_chs)
            frames.append(f)

            if self.__api_mode == API_MODE_SCIENTISST:
                # Get seq number and IO states
                # f.seq = bf[-2] >> 4 | bf[-1] << 4
                f.seq = ((bf[-2] >> 4) | (bf[-1] << 4)) & 0x0FFF

                for i in range(4):
                    f.digital[i] = 0 if (bf[-3] & (0x80 >> i)) == 0 else 1

                # Get channel values
                byte_it = 0
                for i in range(self.__num_chs):
                    index = self.__num_chs - 1 - i
                    curr_ch = self.__chs[index] # outputs 1,2,3,...,8

                    # If it's an AX channel
                    if curr_ch == AX1 or curr_ch == AX2:
                        f.a[index] = (
                            int.from_bytes(
                                bf[byte_it: byte_it + 4], byteorder="little"
                            )
                            & 0xFFFFFF
                        )
                        byte_it += 3
                        if convert:
                            f.mv[index] = (
                                (f.a[index]) * (3.3*2) / (pow(2, 24) - 1))*1000
                            f.mv[index] = round(f.mv[index], 3)

                    # If it's an AI channel
                    else:
                        if not mid_frame_flag:
                            f.a[index] = (
                                int.from_bytes(
                                    bf[byte_it: byte_it + 2], byteorder="little"
                                )
                                & 0xFFF
                            )
                            byte_it += 1
                            mid_frame_flag = 1
                        else:
                            f.a[index] = (
                                int.from_bytes(
                                    bf[byte_it: byte_it + 2], byteorder="little"
                                )
                                >> 4
                            )
                            byte_it += 2
                            mid_frame_flag = 0

                        # Add this workaround to replace AI2 values (ignore) with DAC values
                        if curr_ch == AI2:
                            f.a[index] = curr_dac_value

                        if convert:
                            f.mv[index] = self.__adc1_chars.esp_adc_cal_raw_to_voltage(
                                f.a[index]
                            )
            elif self.__api_mode == API_MODE_SCIENTISST_V2:
                # Get timestamp (us) and IO states
                f.seq = (bf[-1] << 28) | (bf[-2] << 20) | (bf[-3] <<
                                                           12) | (bf[-4] << 4) | ((bf[-5] & 0xF0) >> 4)
                for i in range(4):
                    f.digital[i] = 0 if (bf[-6] & (0x80 >> i)) == 0 else 1

                # Get channel values
                byte_it = 0
                for i in range(self.__num_chs):
                    index = self.__num_chs - 1 - i
                    curr_ch = self.__chs[index]

                    # If it's an AX channel
                    if curr_ch == AX1 or curr_ch == AX2:
                        f.a[index] = (
                            int.from_bytes(
                                bf[byte_it: byte_it + 4], byteorder="little"
                            )
                            & 0xFFFFFF
                        )
                        byte_it += 3
                        if convert:
                            f.mv[index] = (
                                (f.a[index]) * (3.3*2) / (pow(2, 24) - 1))*1000
                            f.mv[index] = round(f.mv[index], 3)

                    # If it's an AI channel
                    else:
                        if not mid_frame_flag:
                            f.a[index] = (
                                int.from_bytes(
                                    bf[byte_it: byte_it + 2], byteorder="little"
                                )
                                & 0xFFF
                            )
                            byte_it += 1
                            mid_frame_flag = 1
                        else:
                            f.a[index] = (
                                int.from_bytes(
                                    bf[byte_it: byte_it + 2], byteorder="little"
                                )
                                >> 4
                            )
                            byte_it += 2
                            mid_frame_flag = 0

                        # Add this workaround to replace AI2 values (ignore) with DAC values
                        if curr_ch == AI2:
                            f.a[index] = curr_dac_value

                        if convert:
                            f.mv[index] = self.__adc1_chars.esp_adc_cal_raw_to_voltage(
                                f.a[index]
                            )
            elif self.__api_mode == API_MODE_JSON:
                print(bf)
            else:
                raise NotSupportedError()

            start += self.__packet_size

        if len(frames) == self.__num_frames:
            if not matrix:
                return frames
            else:
                return np.array([frame.to_matrix() for frame in frames])
        else:
            raise ContactingDeviceError()

    def stop(self):
        """
        Stops a signal acquisition.

        Raises:
            DeviceNotInAcquisitionError: If the device is not in acquisition mode.
        """
        if self.__num_chs == 0:
            raise DeviceNotInAcquisitionError()

        cmd = b"\x00"
        self.__send(cmd)  # 0  0  0  0  0  0  0  0 - Go to idle mode

        self.__num_chs = 0
        self.__sample_rate = 0

        # Cleanup existing data in bluetooth socket
        self.__clear()

    def battery(self, value=0):
        """
        Sets the battery voltage threshold for the low-battery LED.

        Args:
            value (int): Battery voltage threshold. Default value is 0.

                Value | Voltage Threshold
                ----- | -----------------
                    0 |   3.4 V
                 ...  |   ...
                   63 |   3.8 V

        Raises:
            DeviceNotIdleError: If the device is in acquisition mode.
            InvalidParameterError: If an invalid battery threshold value is given.
        """
        if self.__num_chs != 0:
            raise DeviceNotIdleError()

        if value < 0 or value > 63:
            raise InvalidParameterError()

        cmd = value << 2
        # <bat threshold> 0 0 - Set battery threshold
        self.__send(cmd)

    def trigger(self, digital_output):
        """
        Assigns the digital outputs states.

        Args:
            digital_output (list): Vector of booleans to assign to digital outputs, starting at first output (O1).

        Raises:
            InvalidParameterError: If the length of the digital_output array is different from 2.
        """
        length = len(digital_output)

        if length != 2:
            raise InvalidParameterError()

        cmd = 0xB3  # 1  0  1  1  O2 O1 1  1 - Set digital outputs

        for i in range(length):
            if digital_output[i]:
                cmd |= 0b100 << i

        self.__send(cmd)

    def dac(self, voltage, pwm=False):
        """
        Assigns the analog (DAC) output value (ScientISST 2 only).

        Args:
            voltage: Analog output value to set (0V-3.3V).

        Raises:
            InvalidParameterError: If the voltage value is outside of its range, 0-255.
        """
        if not pwm:
            if voltage < 0 or voltage > 3.3:
                raise InvalidParameterError()

        if pwm:
            if voltage < 0 or voltage > 254:
                raise InvalidParameterError()

        cmd = 0xA3  # 1  0  1  0  0  0  1  1 - Set dac output

        if not pwm:
            # Convert from voltage to raw:
            raw = int(voltage * 255 / 3.3)

        if pwm:
            raw = int(voltage)

        cmd |= raw << 8
        self.__send(cmd, nrOfBytes=2)

    # TODO: test with ScientISST Sense v2
    def state(self):
        """
        Returns current device state (%ScientISST 2 only).

        Returns:
            state (State): Current device [`State`][scientisst.state.State]

        Raises:
            DeviceNotIdleError: If the device is in acquisition mode.
            ContactingDeviceError: If there is an error contacting the device.
        """
        if self.__num_chs != 0:
            raise DeviceNotIdleError()

        cmd = 0x0B
        self.__send(cmd)
        # 0  0  0  0  1  0  1  1 - Send device status

        # if (recv(&statex, sizeof statex) != sizeof statex)    # a timeout has occurred
        # throw Exception(Exception::CONTACTING_DEVICE);
        result = self.__recv(16)
        if not result or not self.__checkCRC4(result, 16):
            raise ContactingDeviceError()

        state = State()
        print(result)

        # for(int i = 0; i < 6; i++)
        # state.analog[i] = statex.analog[i];

        # state.battery = statex.battery;
        # state.batThreshold = statex.batThreshold;

        # for(int i = 0; i < 4; i++)
        # state.digital[i] = ((statex.portsCRC & (0x80 >> i)) != 0);

        # return state;

    def disconnect(self):
        """
        Disconnects from a ScientISST device. If an aquisition is running, it is stopped
        """
        if self.__num_chs != 0:
            self.stop()
        if self.__socket:
            self.__socket.shutdown(socket.SHUT_RDWR)
            self.__socket.close()
            self.__socket = None
        elif self.__serial:
            self.__serial.close()
            self.__serial = None
        sys.stdout.write("Disconnected\n")

    def __setupSocket(self):
        """
        Create a socket in function of the comunication mode desired
        """
        if self.com_mode == COM_MODE_BT:
            sys.stdout.write("Connecting to {}...\n".format(self.address))
            # Create the client socket
            if sys.platform == "linux":
                # Check if address is a valid bt MAC address
                if not re.match(
                    "[0-9a-f]{2}([-:]?)[0-9a-f]{2}(\\1[0-9a-f]{2}){4}$",
                    self.address.lower(),
                ):
                    raise InvalidAddressError()

                self.__socket = socket.socket(
                    socket.AF_BLUETOOTH, socket.SOCK_STREAM, socket.BTPROTO_RFCOMM
                )
                self.__socket.connect((self.address, 1))
                self.__socket.settimeout(TIMEOUT_IN_SECONDS)
            else:
                self.__serial = serial.Serial(
                    self.address, self.serial_speed, timeout=TIMEOUT_IN_SECONDS
                )
        elif self.com_mode == COM_MODE_SERIAL:
            self.__serial = serial.Serial(
                self.address, self.serial_speed, timeout=TIMEOUT_IN_SECONDS
            )
        elif self.com_mode == COM_MODE_TCP_SERVER:
            if not self.address.isdigit():
                raise InvalidAddressError()

            port = int(self.address)

            with socket.socket() as s:
                s.bind(("", port))
                print("Binded port %d on all interfaces" % (port))

                s.listen(5)
                print("TCP Server created. Waiting for ScientISST to connect...")

                self.__socket, addr = s.accept()
                print("ScientISST with address", addr, " connected")

        elif self.com_mode == COM_MODE_TCP_AP:
            if isinstance(self.address, str):
                if not self.address.isdigit():
                    raise InvalidAddressError()
                port = int(self.address)
            elif isinstance(self.address, int):
                port = self.address
            else:
                raise InvalidAddressError()

            self.__socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.__socket.connect(("scientisst.local", port))

        else:
            raise InvalidParameterError

    def __getPacketSize(self):
        packet_size = 0

        if self.__api_mode == API_MODE_SCIENTISST:
            num_intern_active_chs = 0
            num_extern_active_chs = 0

            for ch in self.__chs:
                if ch:
                    # Add 24bit channel's contributuion to packet size
                    if ch == AX1 or ch == AX2:
                        num_extern_active_chs += 1
                    # Count 12bit channels
                    else:
                        num_intern_active_chs += 1

            # Add 24bit channel's contributuion to packet size
            packet_size = 3 * num_extern_active_chs

            # Add 12bit channel's contributuion to packet size
            if not (num_intern_active_chs % 2):  # If it's an even number
                packet_size += (num_intern_active_chs * 12) / 8
            else:
                packet_size += (
                    (num_intern_active_chs * 12) - 4
                ) / 8  # -4 because 4 bits can go in the I/0 byte
            # for the I/Os and seq+crc bytes
            packet_size += 3
        elif self.__api_mode == API_MODE_SCIENTISST_V2:
            num_intern_active_chs = 0
            num_extern_active_chs = 0

            for ch in self.__chs:
                if ch:
                    # Add 24bit channel's contributuion to packet size
                    if ch == AX1 or ch == AX2:
                        num_extern_active_chs += 1
                    # Count 12bit channels
                    else:
                        num_intern_active_chs += 1

            # Add 24bit channel's contributuion to packet size
            packet_size = 3 * num_extern_active_chs

            # Add 12bit channel's contributuion to packet size
            if not (num_intern_active_chs % 2):  # If it's an even number
                packet_size += (num_intern_active_chs * 12) / 8
            else:
                packet_size += (
                    (num_intern_active_chs * 12) - 4
                ) / 8  # -4 because 4 bits can go in the I/0 byte
            # for the I/Os and seq+crc bytes
            packet_size += 6
        else:
            raise NotSupportedError()

        return int(packet_size)

    def __changeAPI(self, api):
        if self.__num_chs and self.__num_chs != 0:
            raise DeviceNotIdleError()

        if api <= 0 or api > 3 and api != 14:
            raise InvalidParameterError()

        self.__api_mode = api

        api <<= 4
        api |= 0b11

        self.__send(api)

    def __checkCRC4(self, data, length):
        CRC4tab = [0, 3, 6, 5, 12, 15, 10, 9, 11, 8, 13, 14, 7, 4, 1, 2]
        crc = 0
        if self.__api_mode == API_MODE_SCIENTISST_V2:
            for i in range(length - 5):
                b = data[i]
                crc = CRC4tab[crc] ^ (b >> 4)
                crc = CRC4tab[crc] ^ (b & 0x0F)
            crc = CRC4tab[crc] ^ (data[-5] >> 4)  # First 4 bits
            crc = CRC4tab[crc] ^ (data[-4] >> 4)
            crc = CRC4tab[crc] ^ (data[-4] & 0x0F)
            crc = CRC4tab[crc] ^ (data[-3] >> 4)
            crc = CRC4tab[crc] ^ (data[-3] & 0x0F)
            crc = CRC4tab[crc] ^ (data[-2] >> 4)
            crc = CRC4tab[crc] ^ (data[-2] & 0x0F)
            crc = CRC4tab[crc] ^ (data[-1] >> 4)
            crc = CRC4tab[crc] ^ (data[-1] & 0x0F)

            crc = CRC4tab[crc]

            return crc == (data[-5] & 0x0F)

        else:
            for i in range(length - 2):
                b = data[i]
                crc = CRC4tab[crc] ^ (b >> 4)
                crc = CRC4tab[crc] ^ (b & 0x0F)
            # CRC for seq
            crc = CRC4tab[crc] ^ (data[-2] >> 4)  # First 4 bits

            # Last 8 bits of seq
            crc = CRC4tab[crc] ^ (data[-1] >> 4)
            crc = CRC4tab[crc] ^ (data[-1] & 0x0F)

            crc = CRC4tab[crc]

            return crc == (data[-2] & 0x0F)

    def __send(self, command, nrOfBytes=0):
        """
        Send data
        """

        if nrOfBytes <= 4:
            nrOfBytes = 4
        else:
            raise ValueError("Maximum send command size is 4 bytes")

        if type(command) is int:
            if command != 0:
                command = command.to_bytes(
                    int(log2(command) // 8 + 1), byteorder="little"
                )
            else:
                command = b"\x00"
        if nrOfBytes and len(command) < nrOfBytes:
            for _ in range(nrOfBytes - len(command)):
                command += b"\x00"
        # if self.__serial:
        time.sleep(0.250)
        if self.__log:
            sys.stdout.write(
                "{} bytes sent: {}\n".format(
                    len(command), " ".join("{:02x}".format(c) for c in command)
                )
            )
        if self.__socket:
            self.__socket.send(command)
        elif self.__serial:
            self.__serial.write(command)
        else:
            raise InvalidParameterError()
        # else:
        # raise ContactingDeviceError()

    def __recv(self, nrOfBytes, waitall_flag=True):
        """
        Receive data
        """
        result = b""
        if self.__socket:
            # We have to use a select here with a single socket because we can't apply a timeout in any other way
            ready = select.select([self.__socket], [],
                                  [], 5)  # 10 seconds timeout
            if ready[0]:
                if waitall_flag:
                    remaining = nrOfBytes
                    while remaining > 0:
                        ready = select.select([self.__socket], [], [], 10)
                        if not ready[0]:
                            raise ContactingDeviceError()
                        temp = self.__socket.recv(remaining)
                        result += temp
                        remaining = nrOfBytes - len(result)
                else:
                    result = self.__socket.recv(nrOfBytes)
        elif self.__serial:
            result = self.__serial.read(nrOfBytes)
        else:
            raise InvalidParameterError()
        if self.__log:
            if nrOfBytes > 1:
                sys.stdout.write(
                    "{} bytes received: {}\n".format(
                        nrOfBytes, " ".join("{:02x}".format(c) for c in result)
                    )
                )
            else:
                sys.stdout.write(
                    "{} bytes received: {}\n".format(1, result.hex()))
        return result

    def __clear(self):
        """
        Clear the device buffer
        """
        if self.__socket:
            self.__socket.setblocking(False)
        elif self.__serial:
            self.__serial.timeout = 0
        else:
            raise InvalidParameterError()

        try:
            while self.__recv(1):
                pass
        except BlockingIOError:
            pass

        if self.__socket:
            self.__socket.setblocking(True)
        elif self.__serial:
            self.__serial.timeout = TIMEOUT_IN_SECONDS
        else:
            raise InvalidParameterError()
