import sys
from argparse import ArgumentParser, RawTextHelpFormatter
from scientisst.constants import *


class ArgParser:
    class MyParser(ArgumentParser):
        def error(self, message):
            sys.stderr.write("error: %s\n\n" % message)
            self.print_help()
            sys.exit(2)

    def __init__(self):
        usage = "%(prog)s [args] address"
        description = "description: The program connects to the ScientISST Sense device and starts an acquisition, providing the option to store the received data in a .csv file."
        self.parser = self.MyParser(
            usage=usage, description=description, formatter_class=RawTextHelpFormatter
        )

        self.parser.add_argument(
            "address",
            nargs="?",
            type=str,
            default='/dev/tty.ScientISST-3A6', #COM17
            help="For BTH communication:\n\tLinux: BTH MAC address\n\tMac: serial port address\n\tWindows: BTH serial COM port\nFor TCP/UDP communication:\n\tAll plataforms: server port.",
        )
        self.parser.add_argument(
            "-f",
            "--frequency",
            dest="fs",
            help="sampling frequency, default: 1000",
            type=int,
            default=1000,
        )
        self.parser.add_argument(
            "-c",
            "--channels",
            dest="channels",
            type=str,
            help="analog channels, default: 1,2,3,4,5,6",
            default="1,2,3,4,6,7",
        )
        self.parser.add_argument(
            "-d",
            "--duration",
            dest="duration",
            help="duration in seconds, default: unlimited",
            type=int,
            default=200,
        )
        self.parser.add_argument(
            "-o",
            "--output",
            dest="output",
            help="write report to output file, default: None",
            type=str,
            default="output.csv",
        )
        self.parser.add_argument(
            "-r",
            "--raw",
            action="store_false",
            dest="convert",
            default=False,
            help="do not convert from raw to mV",
        )
        self.parser.add_argument(
            "-s",
            "--lsl",
            dest="stream",
            action="store_true",
            default=True,
            help="stream data using Lab Streaming Layer protocol. Use `python -m pylsl.examples.ReceiveAndPlot` to view stream",
        )
        self.parser.add_argument(
            "--script",
            dest="script",
            help="send the received frames to a script that inherits the CustomScript class",
            type=str,
            default=None,
        )
        self.parser.add_argument(
            "-q",
            "--quiet",
            action="store_false",
            dest="verbose",
            default=True,
            help="don't print ScientISST frames",
        )
        self.parser.add_argument(
            "-v",
            "--version",
            dest="version",
            action="store_true",
            default=False,
            help="show sense.py version",
        )
        self.parser.add_argument(
            "--verbose",
            dest="log",
            action="store_true",
            default=False,
            help="log sent/received bytes",
        )
        self.parser.add_argument(
            "-m",
            "--mode",
            dest="mode",
            type=str,
            default=COM_MODE_BT,
            help="The communication mode. Currently supported modes: "
            + ", ".join(COM_MODE_LIST)
            + ". Default: "
            + COM_MODE_BT,
        )
        self.args = self.parser.parse_args()

    def error(self, value):
        self.parser.error(value)
