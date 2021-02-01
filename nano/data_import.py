# the Nano module

"""
`nano`'s `data_import`
===

"""

import os
import re
import numpy as np
import pandas as pd
import os.path as op

global PATH, DATA, X, Y, XNAME, YNAME, PATHS

PATH = "."
PATHS = {}
DATA = {}
X = None
Y = None
XNAME = ""
YNAME = ""
FILES = []
FILTER = slice(0, None)


def read_dir(ext="", sorting=lambda nm: nm):
    global PATH, DATA, FILES, PATHS
    try:
        PATH, _dir_, files = next(os.walk(PATH))
        FILES += files
        PATHS.update(dict(zip(files, [PATH for f in files])))
        if len(FILES) < 1:
            dir_str = "\n\t".join(_dir_)
            fls_str = "\n\t".join(files)
            print("No files found. Directory content:\n"
                  f"Files:\n\t{fls_str}\n"
                  f"Dirs:\n\t{dir_str}")
    except StopIteration as e:
        if op.isfile(PATH):
            fname = op.basename(PATH)
            FILES += [fname]
            PATHS[fname] = op.dirname(PATH)
    FILES = sorted(FILES, key=sorting)
    sep = r"(\t{1,}|,|;)"
    for fname in FILES:
        if ext in fname:
            dat = pd.read_csv(op.join(PATHS[fname], fname), sep=sep, header=0,
                              engine='python')
            dat_cols = list(dat.columns)
            if len(dat_cols) > 1:
                dat = dat.drop(columns=dat_cols[1::2])
            DATA[fname] = dat
    return len(DATA)


def get_data(file_name=None, file_no=0):
    global X, Y, XNAME, YNAME, DATA, FILES, FILTER
    try:
        if file_name is not None:
            X = np.array(DATA[file_name][XNAME])[FILTER]
            Y = np.array(DATA[file_name][YNAME])[FILTER]
        elif len(DATA) > 0 and len(FILES) > 0:
            X = np.array(DATA[FILES[file_no]][XNAME])[FILTER]
            Y = np.array(DATA[FILES[file_no]][YNAME])[FILTER]
        else:
            raise IndexError(f"No data to get, DATA is {DATA}")
    except IndexError:
        if len(FILES) > 0:
            raise ValueError(f"Wrong data file number: {file_no};"
                             f" min: 0; max: {len(FILES)}")
        else:
            raise FileNotFoundError(f"Cannot read from nonexistent file.")
    except KeyError:
        raise NameError(f"One of XNAME: {XNAME}, YNAME: {YNAME} is invalid or"
                        " not specified.")
    return X, Y
