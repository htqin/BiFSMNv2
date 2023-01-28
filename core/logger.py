from datetime import datetime
from enum import IntEnum
from enum import unique
import inspect
import sys
import logging
logger = logging.getLogger(__name__)

__all__ = ['Logger', 'Verbose', 'set_slim_logger_level']


@unique
class Verbose(IntEnum):
    """
    verbose enum
    """
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    FATAL = 4

    def describe(self):
        """
        Usage: verbose.WARNING.describe()  --> ('WARNING', 1)
        """
        return self.name, self.value

    def __str__(self):
        """
        Usage: str(verbose.WARNING)  --> 'WARNING'
        """
        return 'Target verbose is {0}'.format(self.name)

    @staticmethod
    def default_verbose():
        return Verbose.INFO


'''
verbose to msg dict.
'''
VERBOSE_DICT = {
    'DEBUG': 'DEBUG',
    'INFO': 'INF',
    'WARNING': 'WAR',
    'ERROR': 'ERR',
    'FATAL': 'FTL'
}


class ShellColors:
    """
    shell color decorator class
    """
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    ERROR = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

    def __init__(self, func):
        self.func = func

    def __call__(self, *args):
        if args[1] == Verbose.DEBUG:
            return self.func(args[0], args[1]) + self.ENDC
        elif args[1] == Verbose.INFO:
            return self.BOLD + self.func(args[0], args[1]) + self.ENDC
        elif args[1] == Verbose.WARNING:
            return self.WARNING + self.func(args[0], args[1]) + self.ENDC
        elif args[1] == Verbose.ERROR:
            return self.ERROR + self.func(args[0], args[1]) + self.ENDC
        elif args[1] == Verbose.FATAL:
            return self.ERROR + self.BOLD + self.UNDERLINE + self.func(
                args[0], args[1]) + self.ENDC
        else:
            raise NameError('ERROR: Logger not support verbose: %s' %
                            (str(args[1])))


@ShellColors
def with_color(header, verbose):
    return header


logger_level = Verbose.INFO


class Logger:
    """
    Logger class.
    """
    WithColor = True
    # lock mutex for writing log files by multi process.
    # lock = Lock()

    Prune = lambda filename: filename.split('/')[-1]

    def __init__(self, verbose=Verbose.default_verbose()):
        """
        """
        self.__verbose = verbose
        self.log_head = "" + VERBOSE_DICT[verbose.name] + " | " + str(
            datetime.now()) + " | "
        # 1 represents line at caller
        callerframerecord = inspect.stack()[1]
        frame = callerframerecord[0]
        info = inspect.getframeinfo(frame)
        Prune = lambda filename: filename.split('/')[-1]
        self.log_head += Prune(info.filename) + ":" + str(
            info.lineno) + " " + str(info.function) + "() ] "

    def __call__(self, *args, **kwargs):
        """
        feed info to log engine.
        """
        msg = ''.join(str(i) for i in args)
        full_msg = ""
        try:
            full_msg = (with_color(self.log_head, self.__verbose) + " " + msg) \
                if Logger.WithColor else (self.log_head + " " + msg)
        except NameError:
            raise
        self.log_to_everywhere(full_msg)

    def log_to_everywhere(self, full_msg):
        """
        log to stdout and file
        """
        if int(logger_level) <= int(self.__verbose):
            print(full_msg)

        # Logger.lock.acquire()
        # log to stdout

        sys.stdout.flush()
        # logger.lock.release()


def set_slim_logger_level(level: Verbose):
    """
    Args:
        level: set logger level for stdout, support: 'info', 'warning', 'error', 'fatal'

    Returns:

    """
    global logger_level
    if int(level) <= int(Verbose.FATAL):
        logger_level = level
    else:
        raise RuntimeError('unsupported logging level: ', level)
