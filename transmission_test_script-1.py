#!/usr/bin/python

"""
sense.py
"""
import logging
import sys
import traceback
from scientisst import *
from scientisst import __version__
from threading import Timer
from threading import Event
from sense_src.arg_parser import ArgParser
from sense_src.custom_script import get_custom_script, CustomScript
from sense_src.device_picker import DevicePicker
from sense_src.file_writer import *
from math import ceil
import numpy as np
import pandas as pd
import os
import sys


def run_scheduled_task(duration, stop_event):
    def stop(stop_event):
        stop_event.set()

    timer = Timer(duration, stop, [stop_event])
    timer.start()
    return timer


# saving path
root_path = sys.argv[0]
main_dir, _ = os.path.split(root_path)
main_saving_path = os.path.join(main_dir, 'results')

test_folder_name = 'test-1'
test_path = os.path.join(main_saving_path, test_folder_name)

# create results folder
if not os.path.exists(test_path):
    os.makedirs(test_path)

# logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# TODO: variables to edit for `test-1`
# TODO: Duration, sampling frequencies (fs_l), Nr. analog channels (ch_nrs)
duration = 30  # seconds

# input params to iterate = Sf and Nr Analog CHs
fs_l = [1000 * i for i in range(1, 18, 2)]
ch_nrs_l = [i for i in range(1, 7)]

# compute sent nr of bytes per sample
data_bytes = [ceil(3 + 1.5 * i) for i in range(1, 7)]

output_filenames = []
params_list = []
transfer_rates = []
# transfer rate = bytes / s = bytes * Fs

for sf in fs_l:
    for j in range(1, len(ch_nrs_l) + 1):
        print(ch_nrs_l)
        print(j)

        data_sz = ch_nrs_l[j - 1]
        output_filenames.append("CHs_{}_sfs_{}".format(j, sf))

        analog_param = []
        for i in range(1, j + 1):
            analog_param.append(i)

        params_list.append([sf, analog_param])
        transfer_rates.append(data_sz * sf)

transfer_rates = np.asarray(transfer_rates)

tr_order = np.argsort(transfer_rates)
transfer_rates = transfer_rates[tr_order]

params_list = np.asarray(params_list)

params_list = params_list[tr_order, ...]

# TODO: If the connection stops, ScientISST CORE needs to be restarted manually (OFF/ON).
#  In order to restart the script from the last test, use ID = ID of last saved test in test folder + 1
select_nr = 31


def main():

    for i in range(select_nr, params_list.shape[0]):

        data_rate = transfer_rates[i]
        tmp_sf = params_list[i][0]
        tmp_ch = params_list[i][1]
        print("_" * 10)
        print(tmp_ch)
        print([tmp_sf, tmp_ch])

        # iterate until one successful connection is made for the current parameter combination
        param_connected = False

        while not param_connected:

            try:
                arg_parser = ArgParser()
                args = arg_parser.args

                if args.version:
                    sys.stdout.write("sense.py version {}\n".format(__version__))
                    sys.exit(0)

                if args.address:
                    address = args.address
                else:
                    if args.mode == COM_MODE_BT:
                        address = DevicePicker().select_device()

                        if not address:
                            arg_parser.error("No paired device found")
                    else:
                        arg_parser.error("No address provided")

                args.channels = tmp_ch  # sorted(map(int, args.channels.split(",")))

                scientisst = ScientISST(address, com_mode=args.mode, log=args.log)

                param_connected = True
                output_filename = os.path.join(test_path,
                                               'data_test_{}__{}_{}_{}.txt'.format(i, data_rate, tmp_sf, tmp_ch))

                try:
                    # if args.output:
                    firmware_version = scientisst.version_and_adc_chars(print=False)
                    file_writer = FileWriter(
                        output_filename,
                        address,
                        tmp_sf,  # args.fs,
                        tmp_ch,  # args.channels,
                        args.convert,
                        __version__,
                        firmware_version,
                    )

                    if args.stream:
                        from sense_src.stream_lsl import StreamLSL

                        lsl = StreamLSL(
                            tmp_ch,  # args.channels,
                            tmp_sf,  # args.fs,
                            address,
                        )
                    if args.script:
                        script = get_custom_script(args.script)

                    stop_event = Event()

                    scientisst.start(tmp_sf, tmp_ch)  # args.fs, args.channels)
                    sys.stdout.write("Start acquisition\n")

                    # if args.output:
                    file_writer.start()
                    if args.stream:
                        lsl.start()
                    if args.script:
                        script.start()

                    timer = None
                    if duration > 0:
                        timer = run_scheduled_task(duration, stop_event)
                    try:
                        if args.verbose:
                            header = "\t".join(
                                get_header(tmp_ch, args.convert)) + "\n"  # args.channels, args.convert)) + "\n"
                            sys.stdout.write(header)
                        while not stop_event.is_set():
                            frames = scientisst.read(convert=args.convert)
                            # if args.output:
                            file_writer.put(frames)
                            if args.stream:
                                lsl.put(frames)
                            if args.script:
                                script.put(frames)
                            if args.verbose:
                                sys.stdout.write("{}\n".format(frames[0]))
                    except KeyboardInterrupt:
                        if duration and timer:
                            timer.cancel()
                        pass

                    scientisst.stop()
                    # let the acquisition stop before stoping other threads
                    time.sleep(0.25)

                    sys.stdout.write("Stop acquisition\n")
                    # if args.output:
                    file_writer.stop()

                    if args.stream:
                        lsl.stop()
                    if args.script:
                        script.stop()

                finally:
                    scientisst.disconnect()

            except Exception as error:
                output_filename = os.path.join(test_path,
                                               'data_test_{}__{}_{}_{}.txt'.format(i, data_rate, tmp_sf, tmp_ch))
                log_str = traceback.format_exc()
                with open(output_filename, 'w') as f:
                    f.write(log_str)

                # log errors
                logger.exception(error)


if __name__ == "__main__":
    main()
