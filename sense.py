#!/usr/bin/python

"""
sense.py
"""

import sys
import os
import re
from datetime import datetime
from scientisst import *
from scientisst import __version__
from threading import Timer
from threading import Event
from sense_src.arg_parser import ArgParser
from sense_src.custom_script import get_custom_script, CustomScript
from sense_src.device_picker import DevicePicker
from sense_src.file_writer import *

def run_scheduled_task(duration, stop_event):
    def stop(stop_event):
        stop_event.set()

    timer = Timer(duration, stop, [stop_event])
    timer.start()
    return timer

def main():
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

    args.channels = sorted(map(int, args.channels.split(",")))
    
    device_names = { "COM12": "BE4A", "COM15": "28F2"}
    # build output file path
    if args.output:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        signals_dir = os.path.join(base_dir, "data", "signals")
        os.makedirs(signals_dir, exist_ok=True)
        
        if args.output =="auto":
            port = os.path.basename(address)
            device_name = device_names.get(port.upper(), port)
            device_name = re.sub(r"[^\w\-]", "_", device_name)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename ="{}_{}Hz_{}.csv".format(device_name, args.fs, timestamp)
        else:
            filename = os.path.basename(args.output)
            
        args.output = os.path.join(signals_dir, filename)

    scientisst = ScientISST(address, com_mode=args.mode, log=args.log)

    dac_value = 150  # or any other value you prefer

    try:
        if args.output:
            firmware_version = scientisst.version_and_adc_chars(print=False)
            file_writer = FileWriter(
                args.output,
                address,
                args.fs,
                args.channels,
                args.convert,
                __version__,
                firmware_version,
            )
        if args.stream:
            from sense_src.stream_lsl import StreamLSL

            lsl = StreamLSL(
                args.channels,
                args.fs,
                address,
            )
        if args.script:
            script = get_custom_script(args.script)

        stop_event = Event()

        scientisst.start(args.fs, args.channels)
        sys.stdout.write("Start acquisition\n")

        if args.output:
            file_writer.start()
        if args.stream:
            lsl.start()
        if args.script:
            script.start()

        timer = None
        if args.duration > 0:
            timer = run_scheduled_task(args.duration, stop_event)

        tick = 0

        try:
            if args.verbose:
                header = "\t".join(get_header(args.channels, args.convert)) + "\n"
                sys.stdout.write(header)
            while not stop_event.is_set():

                # add these 2 lines
                frames = scientisst.read(convert=args.convert, curr_dac_value=dac_value)
                dac_value = scientisst.dac_control(dac_value, frames, tick)

                if args.output:
                    file_writer.put(frames)
                if args.stream:
                    lsl.put(frames)
                if args.script:
                    script.put(frames)
                if args.verbose:
                    sys.stdout.write("{}\n".format(frames[0]))

                tick += 1

        except KeyboardInterrupt:
            if args.duration and timer:
                timer.cancel()
            pass

        scientisst.stop()
        # let the acquisition stop before stoping other threads
        time.sleep(0.25)

        sys.stdout.write("Stop acquisition\n")
        if args.output:
            file_writer.stop()
        if args.stream:
            lsl.stop()
        if args.script:
            script.stop()

    finally:
        scientisst.disconnect()

    sys.exit(0)


if __name__ == "__main__":
    main()
