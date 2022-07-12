import logging

from typing import Literal, Sequence

DEFAULT_FORMAT       = "%(asctime)s,%(msecs)d %(levelname)-8s [ %(name)s - %(funcName)s: %(lineno)d ] %(message)s"
DEFAULT_DATEFMT      = "%Y-%m-%d : %H:%M:%S"
DEFAULT_INFO_MODULES = ('datasets', 'matplotlib')


class TextColors:
    ANSI_REGEX    = r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"

    PRE           = "\033["
    RESET         = f"{PRE}0m"

    BLACK         = f"{PRE}0;30m"
    RED           = f"{PRE}0;31m"
    GREEN         = f"{PRE}0;32m"
    YELLOW        = f"{PRE}0;33m"
    BLUE          = f"{PRE}0;34m"
    MAGENTA       = f"{PRE}0;35m"
    CYAN          = f"{PRE}0;36m"
    WHITE         = f"{PRE}0;37m"

    BBLACK        = f"{PRE}1;30m"
    BRED          = f"{PRE}1;31m"
    BGREEN        = f"{PRE}1;32m"
    BYELLOW       = f"{PRE}1;33m"
    BBLUE         = f"{PRE}1;34m"
    BMAGENTA      = f"{PRE}1;35m"
    BCYAN         = f"{PRE}1;36m"
    BWHITE        = f"{PRE}1;37m"


def setup_logging(
    is_quiet        : bool = False,
    level           : Literal[logging._levelToName] = None,
    format          : str = DEFAULT_FORMAT,
    datefmt         : str = DEFAULT_DATEFMT,
    info_modules    : Sequence[str] = DEFAULT_INFO_MODULES
    ):

    if level is None:
        if is_quiet:
            level = logging.INFO
        else:
            level = logging.DEBUG

    logging.basicConfig(level=level, format=format, datefmt=datefmt)
    for module in info_modules:
        logging.getLogger(module).setLevel(logging.INFO)
