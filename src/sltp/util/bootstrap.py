import logging
import argparse
import sys

from .console import get_terminal_size

_LOG_LEVEL = None


class ErrorAbortHandler(logging.StreamHandler):
    """
    Custom logging Handler that exits when a critical error is encountered.
    """
    def emit(self, record):
        logging.StreamHandler.emit(self, record)
        if record.levelno >= logging.CRITICAL:
            sys.exit('aborting')


def setup_logging(level):
    # Python adds a default handler if some log is written before this
    # function is called. We therefore remove all handlers that have
    # been added automatically.
    root_logger = logging.getLogger('')
    for handler in root_logger.handlers:
        root_logger.removeHandler(handler)

    # Handler which writes _LOG_LEVEL messages or higher to stdout
    console = ErrorAbortHandler(sys.stdout)
    # set a format which is simpler for console use
    format_ = '%(asctime)-s %(levelname)-8s %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(format_, datefmt=datefmt)
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    root_logger.addHandler(console)
    root_logger.setLevel(level)


class RawAndDefaultsHelpFormatter(argparse.RawTextHelpFormatter):
    """
    Help message formatter which preserves the description format and adds
    default values to argument help messages.
    """
    def __init__(self, prog, **kwargs):
        # Use the whole terminal width.
        width, _ = get_terminal_size()
        argparse.RawTextHelpFormatter.__init__(self, prog, width=width, **kwargs)

    def _fill_text(self, text, width, indent):
        return '\n'.join([indent + line for line in text.splitlines()])

    def _get_help_string(self, action):
        help = action.help
        if '%(default)' not in action.help and 'default' not in action.help:
            if action.default is not argparse.SUPPRESS:
                defaulting_nargs = [argparse.OPTIONAL, argparse.ZERO_OR_MORE]
                if action.option_strings or action.nargs in defaulting_nargs:
                    help += ' (default: %(default)s)'
        return help


def get_parser(add_log_option=True, **kwargs):
    kwargs.setdefault('formatter_class', RawAndDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(**kwargs)
    if add_log_option:
        parser.add_argument(
            '--log',
            dest='log_level',
            choices=['DEBUG', 'INFO', 'WARNING'],
            default='INFO',
            help='Logging verbosity')
    return parser


def setup_argparser():
    parser = get_parser()
    parser.add_argument('exp_id', metavar='domain:experiment', help="The domain and experiment within that domain.")

    parser.add_argument('--workspace', metavar='dir',
                        help="The directory where the experiment outputs will be left. If none specified, use"
                             " a default directory inside the SLTP project tree.")

    parser.add_argument('--show', action='store_true',
                        help="Show the steps available for the given experiment, but don't run anything.")

    parser.add_argument('steps', metavar='step', nargs='*', default=[], type=int,
                        help='Which of the experiment steps to run (e.g.: "3 4 5" will assume the first two steps were'
                             ' already run and skip them). By default, run them all.')
    return parser


def parse_and_set_log_level():
    # Set log level only once.
    global _LOG_LEVEL
    if _LOG_LEVEL:
        return

    parser = get_parser(add_help=False)
    args, _ = parser.parse_known_args()

    if getattr(args, 'log_level', None):
        _LOG_LEVEL = getattr(logging, args.log_level.upper())
        setup_logging(_LOG_LEVEL)


parse_and_set_log_level()
