# System
import sys

# Logging
import logging

# Type-hints
from typing import Literal
from . import custom_types

# IO
import json
import gzip
import pickle
from pathlib import Path

import contextlib

logger = logging.getLogger(__name__)


READ_WRITE_LIBS = ("json", "gzip", "pickle")

def determine_read_write_lib(file_path: custom_types.PathType):
    """ Determines library based on file extension. """

    file_path         = Path(file_path)
    file_extension    = file_path.suffix[1:].lower()  # Ignoring the '.' in the beginning

    if file_extension == "json":
        lib = "json"
    # elif file_extension in ("pt", "pth"):
    #     lib = "torch"
    elif file_extension == "gz":
        lib = "gzip"
    elif file_extension == "p":
        lib = "pickle"
    else:
        raise ValueError(f"Unsupported file path extension for: '{file_path}'")

    return lib

def determine_read_write_funcs(lib: Literal[READ_WRITE_LIBS], is_read: bool = True):
    """ Determines read/write (save/load) functions based on a library. """

    open_func   = open if lib != "gzip" else gzip.open
    mode        = 'r' if is_read else 'w'
    is_binary   = False

    if lib == "json":
        save_or_load_func = json.load if is_read else json.dump
        is_binary = is_read

    else:
        if not is_read:
            is_binary = True

        # if lib == "torch":
        #     save_or_load_func = torch.load if is_read else torch.save
        #     is_binary = True

        if lib in ("gzip", "pickle"):
            save_or_load_func = pickle.load if is_read else pickle.dump
            is_binary = True

        else:
            raise NotImplementedError(f"Bad library given: '{lib}'. Should be any of the following: {READ_WRITE_LIBS}.")

    return open_func, mode + ('b' if is_binary else ''), save_or_load_func

def save_data_by_lib(data, output_path: str = '-', lib: Literal[READ_WRITE_LIBS] = None, **kwargs):
    """
    Dumps/Saves data to stdout or output file.

    Args:
        data -          Object to save.
        output_path -   Path to store object to.
        lib -           Library out of: "json", "torch", "gzip", "pickle"
                        (default = None uses output_path extension).
    """

    with contextlib.ExitStack() as stack:
        if output_path == "-":
            fp = sys.stdout
            json.dump(data, fp, indent=2)
        else:
            if lib is None:
                lib = determine_read_write_lib(output_path)

            open_func, mode, save_func    = determine_read_write_funcs(lib, is_read=False)
            fp                            = stack.enter_context(open_func(output_path, mode=mode))
            save_func(data, fp, **kwargs)
            logger.info(f"DONE! Output ready @ '{output_path}'")

def load_data_by_lib(path: custom_types.PathType, lib: Literal[READ_WRITE_LIBS] = None, **kwargs):
    """
    Loads data from input file.

    Args:
        path -          Path to load data from.
        lib -           Library out of: "json", "torch", "gzip", "pickle"
                        (default = None uses output_path extension).
    """

    if lib is None:
        lib = determine_read_write_lib(path)

    open_func, mode, load_func = determine_read_write_funcs(lib, is_read=True)

    with open_func(path, mode=mode) as fp:
        data = load_func(fp, **kwargs)

    return data

def load_config(config_path: custom_types.PathType):
    """ Loads configuration data from file. """

    logger.debug(f"Loading configuration from '{config_path}'...")
    return load_data_by_lib(config_path)
