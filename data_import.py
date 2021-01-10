import os
import re
import numpy as np
import pandas as pd
import os.path as op

global PATH, DATA, X, Y, XNAME, YNAME

PATH = "."
DATA = {}
X = None
Y = None
XNAME = ""
YNAME = ""
FILES = []


def read_dir(ext="", sorting=lambda nm: nm):
    global PATH, DATA, FILES
    PATH, dir, FILES = next(os.walk(PATH))
    FILES = sorted(FILES, key=sorting)
    for fname in FILES:
        if ext in fname:
            DATA[fname] = pd.read_csv(op.join(PATH, fname), sep=r"\t{1,}",
                                      header=0, engine='python')
    return len(DATA)


def get_data(file_name=None, file_no=0):
    global X, Y, XNAME, YNAME, DATA, FILES
    try:
        if file_name is not None:
            X = np.array(DATA[file_name][XNAME])
            Y = np.array(DATA[file_name][YNAME])
        else:
            X = np.array(DATA[FILES[file_no]][XNAME])
            Y = np.array(DATA[FILES[file_no]][YNAME])
    except IndexError:
        raise ValueError(f"Wrong data file number: {file_no}; min: 0; max: 36")
    return X, Y
