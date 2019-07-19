"""
DataStruct Module
---

Description:
---
This module have been created to collect all methods, that were needed to
manage and modify data, especially spectral. It contains some statistical
methods but also analitical ones.

To effectively manage data files there are listing and converting functions.
Most of them created to be quite easy to use and error proof, although
all of these can be broken. Try not to do it.
"""

import csv
import os

global communicate

communicate = (("Warning: last element of derivative iterable is an"
                " arithmetic mean of all other datapoints by default.\n"),
               ("Warning: last element of derivative iterable is an"
                " geometric mean of all other datapoints by default.\n"))


# HELPER CLASS


class Helper:
    """
    `DataStruct` Helper class
    ---
    Description:
    ---
    This class contains helper functions that are not directly needed for
    `DataStruct` but are used by this class.
    """
    @staticmethod
    def gravity_mean(data, func=lambda x: 1/(x+1)**2):
        """
        `DataStruct` helper function
        ---
        Description:
        ---
        Calculates gravitational mean of elements in `data`.

        Raises:
        ---
        `TypeError: Argument is not a valid iterable.` if argument is not a
        type like `tuple`, `list` or `set`.
        """
        try:
            data = tuple(data)
            __g = [func(n) for n in range(len(data))]
            return sum([data[n]*__g[n]/sum(__g) for n in range(len(data))])
        except Exception:
            raise TypeError("Argument is not a valid numerical iterable.")

    @staticmethod
    def arith_mean(data):
        """
        `DataStruct` helper function
        ---
        Description:
        ---
        Calculates arithmetic mean of elements in `data`.

        Raises:
        ---
        `TypeError: Argument is not a valid iterable.` if argument is not a
        type like `tuple`, `list` or `set`.
        """
        try:
            data = tuple(data)
            return sum(data)/len(data)
        except Exception:
            raise TypeError("Argument is not a valid numerical iterable.")

    @staticmethod
    def geom_mean(data, func=lambda x: x):
        """
        `DataStruct` helper function
        ---
        Description:
        ---
        Calculates geometric mean of elements in `data`.

        Raises:
        ---
        `TypeError: Argument is not a valid iterable.` if argument is not a
        type like `tuple`, `list` or `set`.
        """
        try:
            data = tuple(data)
            dat_len = 1/len(data)
            result = 1
            for dat in data:
                result *= (func(dat)**dat_len)
            return result
        except Exception:
            raise TypeError("Argument is not a valid numerical iterable.")

    @staticmethod
    def sigma(data):
        """
        `DataStruct` helper function
        ---
        Description:
        ---
        Calculates standard deviation of elements in `data`.

        Raises:
        ---
        `TypeError: Argument is not a valid iterable.` if argument is not a
        type like `tuple`, `list` or `set`.
        """
        try:
            data = tuple(data)
            av = Helper.arith_mean
            data_sq = 0
            for dat in data:
                data_sq += dat**2
            return (data_sq/len(data) - av(data)**2)**0.5
        except Exception:
            raise TypeError("Argument is not a valid numerical iterable.")

    @staticmethod
    def cov(data_x, data_y, estimator="arithmetic"):
        """
        `DataStruct` helper function
        ---
        Description:
        ---
        Calculates covariance of elements in `data_x` and `data_y`.

        Raises:
        ---
        +   `TypeError: Arguments are not a valid numerical iterable.` if
        argument is not a type like `tuple`, `list` or `set`.

        +   `ValueError: Wrong estimator specified` if `estimator` is not
        `arithmetic` or `geometric`.

        Mods:
        ---
        This function conatains two modes for calculating an estimator:
        arithmetic or geometric. First one is a default, second can be raised
        by modifying `estimator` string argument by passing `geometric` to it.
        """
        try:
            if estimator == "arithmetic":
                mean = Helper.arith_mean
            elif estimator == "geometric":
                mean = Helper.geom_mean
            else:
                raise ValueError("Wrong estimator specified: {est_name}."
                                 "".format(est_name=mean.__name__))
            data_x = tuple(data_x)
            data_y = tuple(data_y)
            sum_result = ()
            for n in range(len(data_x)):
                sum_result += (data_x[n]*data_y[n],)
            return mean(sum_result) - mean(data_x)*mean(data_y)
        except Exception:
            TypeError("Arguments are not a valid numerical iterable.")

    @staticmethod
    def sigmer(str_val):
        """
        `DataStruct` helper function
        ---
        Description:
        ---
        This method returns precision of floating point values to use it as
        exponent of 10 for stddev-precision solution.

        Parameters:
        ---
        +   `value` - floating point value

        Returns:
        ---
        Tuple containing precision and scale of passed argument.
        """
        # for v in str_val.split("."):
        #     pass
        # prec = len(str_val.split(".")[0]) + len(str_val.split(".")[1])
        print("Funkcja niegotowa do użytku.")


# MAIN CLASS

class DataStruct(Helper):
    """
    Description:
    ---
    Class created to read and manage multiple CSV files in a single directory
    specified as `DataStruct.path` attribute or passed to the constructor
    directly as `DataStruct.files`.

    Requirements:
    ---
    +   `csv` module
    +   `os` module
    """

    def __init__(self, path="", files=[]):
        """
        Description:
        ---
        Initializing function of `DataStruct`.
        """
        self.path = path
        self.files = files
        self.data = []
        self.header = []
        self.derivs = []
        self.integrs = []
        self.real_part = {"name": "", "values": []}
        self.imag_part = {"name": "", "values": []}

    def get_files(self):
        """
        Description:
        ---
        Method reads CSV files included in `DataStruct.path` directory and
        stores data about them in a list. Prepares data to be extracted
        by `DataStruct.load_data()` method.

        Raises:
        ---
        Warning: `Path: {path} contains no CSV files to load.` if path is
        out of CSV files.
        """
        files = listing(self.path)['files']
        self.files = []
        for file in files:
            if '.csv' in file:
                self.files.append(os.path.join(self.path, file))
        if self.files == []:
            print("Warning: Path: {path} contains"
                  " no CSV files to load.".format(path=self.path))

    def load_data(self):
        """
        Description:
        ---
        Extracting data from `DataStruct.files` read by using `csv` module.
        Separates headers of CSV files and stores them in
        `DataStruct.headers`.
        """
        for n in range(len(self.files)):
            csv_reader = csv.reader(open(self.files[n]))
            self.header = next(csv_reader)
            self.data.append({})
            for k in range(len(self.header)):
                self.data[n][self.header[k]] = []
            for row in csv_reader:
                for k in range(len(self.header)):
                    self.data[n][self.header[k]].append(float(row[k]))

    def normalize(self, function, value=1):
        """
        Description:
        ---
        Normalizes data stored in object by performing
        calculation:
        data[n] /= function(data)

        Parameters:
        ---
        +   `function` - parameter of function
        +   `value` - parameter of const by which data is multiplied
        """
        if float(value) != value:
            raise TypeError("""Value argument: '{val}' must be of
            numerical value.""".format(val=value))
        if function is None:
            function = sum
        for data in self.data:
            for dat in data:
                constans = function(data[dat])/value
                if dat in self.imag_part or self.imag_part == []:
                    for n in range(len(data[dat])):
                        data[dat][n] /= constans

    def separation(self):
        pass
        # if
        # for dat in self.data:
        #     self.real_part["values"].append(tuple(dat[R_header]))
        #     self.imag_part["values"].append(tuple(dat[I_header]))
        # self.real_part['name'] = R_header
        # self.imag_part['name'] = I_header

    def read_routine(self):
        self.__init__(self.path)
        self.get_files()
        self.load_data()
        self.separation()


# MODULE FUNCTIONS

def deriv(x_arg, y_arg, func=Helper.gravity_mean):
    """
    Description:
    ---
    Returns derivative of `y_arg` over `x_arg`.

    Parameters:
    ---
    +   `x_arg` Real part of data [iterable]
    +   `y_arg` Imag part of data [iterable]
    Both parameters must be of the same lenght; if not `IndexError`
    is raised.

    Returns:
    ---
    List of derivative's data points of the same length as parameters'
    length.

    Raises:
    ---
    +   `IndexError: Parameters are not of same length ({len(x_arg)} and
    {len(y_arg)}).` if parameters are not of equal length.
    +   `ValueError: Parameter {arg} is too short: {len(arg)}.` if
    parameter arg is not at least 2 elements long.
    """
    if len(x_arg) != len(y_arg):
        raise IndexError("Parameters are not of same length ({lenx} "
                         "and {leny}).".format(lenx=len(x_arg),
                                               leny=len(y_arg)))
    if len(x_arg) < 2:
        raise ValueError("Parameter x_arg is too short: "
                         "{lenx}".format(lenx=len(x_arg)))
    if len(y_arg) < 2:
        raise ValueError("Parameter y_arg is too short: "
                         "{lenx}.".format(lenx=len(x_arg)))
    dydx = ()
    for n in range(len(x_arg)-1):
        dy = y_arg[n]-y_arg[n+1]
        dx = x_arg[n]-x_arg[n+1]
        dydx += (dy/dx, )
        c_dydx = list(dydx).copy()
        c_dydx.reverse()
    dydx += (func(c_dydx), )
    print(communicate[0], end="")
    return dydx


def integral(x_arg, y_arg, func=Helper.geom_mean):
    """
    Description:
    ---
    Returns integral of `y_arg` over `x_arg`.

    Parameters:
    ---
    +   `x_arg` Real part of data [iterable]
    +   `y_arg` Imag part of data [iterable]
    Both parameters must be of the same lenght; if not `IndexError`
    is raised.

    Returns:
    ---
    List of integral's data points of the same length as parameters'
    length.

    Raises:
    ---
    +   `IndexError: Parameters are not of same length ({len(x_arg)} and
    {len(y_arg)}).` if parameters are not of equal length.
    +   `ValueError: Parameter {arg} is too short: {len(arg)}.` if
    parameter arg is not at least 2 elements long.
    """
    if len(x_arg) != len(y_arg):
        raise IndexError("Parameters are not of same length ({lenx} "
                         "and {leny}).".format(lenx=len(x_arg),
                                               leny=len(y_arg)))
    if len(x_arg) < 2:
        raise ValueError("Parameter x_arg is too short: "
                         "{lenx}".format(lenx=len(x_arg)))
    if len(y_arg) < 2:
        raise ValueError("Parameter y_arg is too short: "
                         "{lenx}.".format(lenx=len(x_arg)))
    int_ydx = ()
    ydx = ()
    for n in range(len(x_arg)-1):
        dx = x_arg[n+1] - x_arg[n]
        dy = y_arg[n]
        ydx += (dx*dy, )
        int_ydx += (sum(ydx), )
    int_ydx += (func(int_ydx, abs)*dx, )
    print(communicate[1], end="")
    return int_ydx


def nearest(iterable, value, order=0):
    """
    Description:
    ---
    Returns closest value to the specified `value`argument by calculating:
    abs([`iterable`]-`value`)
    and sorting an output. `order` specifies index of
    sorted iterable (n-th closest value).
    """
    abs_val = []
    abs_copy = []
    for it in iterable:
        abs_val.append(abs(it - value))
        abs_copy.append(abs(it - value))
    abs_copy = list(set(abs_copy))
    abs_copy.sort()
    value_found = -1
    return_tuple = ()
    while value_found is not None:
        try:
            value_found = abs_val.index(abs_copy[order], value_found+1)
            return_tuple += (value_found, )
        except ValueError:
            if len(return_tuple) < 2:
                return value_found
            value_found = None
    return return_tuple


def pearson(x_dat, y_dat):
    """
    Description:
    ---
    Pearson correlation coefficient is a method of
    describing correlation between two data sets as
    a single value between `-1` and `1`, where `-1` is a full
    negative correlation and `1` is full positive correlation.
    `0` is representing two non-correlated sets.

    Parameters:
    ---
    +   `x_dat` first data set (iterable of numerical values)
    +   `y_dat` second data set (iterable of numerical values)

    Returns:
    ---
    Pearson correlation coefficient (float).
    """
    if len(x_dat) != len(y_dat):
        raise ValueError(("Parameters are not of equal length:" +
                          "`x_dat` of length {xlen}" +
                          "`y_dat` of length {ylen}").format(xlen=len(x_dat),
                                                             ylen=len(y_dat)))
    covariance = Helper.cov
    sigma = Helper.sigma
    return covariance(x_dat, y_dat)/(sigma(x_dat)*sigma(y_dat))


# MAINTAINANCE FUNCTIONS

def listing(path="."):
    """
    `DataStruct` module method
    ---
    Descritpion:
    ---
    This method returns a `dict` of files and directories in
    specified `path`.

    Parameters:
    ---
    + `path`: str - specifies path to be scanned for files or dirs.

    Raises:
    ---
    + TypeError: Argument `path` is not a vild str.
    + Exception: Path `{path}` is not an existing directory.
    """
    if type(path) != str:
        raise TypeError("Argument `path` is not a valid str.")
    if not os.path.exists(path):
        raise Exception(f"Path `{path}` is not an existing directory.")
    this_dir = {
        "path": "",
        "dirs": [],
        "files": []
    }
    if len(path) > 1:
        if not (path[-1] == '/'):
            path += '/'
    else:
        if path != "/":
            path += "/"

    def ls(path=".", option=""):
        flist = os.popen("ls '"+path+"' "+option).read().split("\n")
        if option.find('g') != -1:
            return [path]+flist[1:]
        else:
            return [path]+flist

    ls_a = ls(path, "-a")
    ls_ag = ls(path, "-ag")
    this_dir["path"] = ls_a[0]
    for i in range(1, len(ls_a)-1):
        if ls_ag[i][0] == "d":
            this_dir["dirs"].append(ls_a[i])
        else:
            this_dir["files"].append(ls_a[i])
    return this_dir


def csv_convert(path=".", file_name="", new_path=".", header=""):
    """
    `DataStruct` module method
    ---

    Description:
    ---
    Converts file in path from column-like data to CSV format.

    Parameters:
    ---
    + `path` - specifies path of file to be converted (if not passed, method
     takes current directory as `path`)
    + `file_name` - specifies name of file to be converted

    Raises:
    ---
    None

    Returns:
    ---
    + 0 - if successful
    + 1 - if `FileNotFoundError` error occurres while trying to
    find and open one of input or output files
    + 2 - if data from input file is not covertable to CSV format
    + -1 - if one or both paths are not str-convertible
    """
    path = str(path)
    new_path = str(new_path)
    if path[-1] != '/':
        path += '/'
    if new_path[-1] != '/':
        new_path += '/'
    try:
        old_file = open(path+file_name, 'r')
        new_csv = open(new_path+file_name.split('.')[0]+'.csv', 'w')
    except FileNotFoundError:
        return 1
    print(header, end="", file=new_csv)
    while True:
        try:
            line = next(old_file).split()
            csv_line = ""
            for word in line:
                csv_line += word+','
            print(csv_line[:len(csv_line)-1], file=new_csv)
        except StopIteration:
            break
        except ValueError:
            print("Something went wrong. Are you sure this file"
                  " can be converted?")
            return 2
    return 0


def csv_manual(path="."):
    """
    `DataStruct` module method
    ---

    Description:
    ---
    Manual searching and converting files from ASCII table format to CSV.

    Parameters:
    ---
    + `path` - specifies path to search for file to be converted
    Raises:
    ---
    None

    Returns:
    ---
    None

    If there is an error occuring it must be traced back to
    `DataStruct.listing` or `DataStruct.csv_convert` methods.
    """
    file_list = listing(path)['files']
    pos = 1
    print("Choose file to convert:")
    for f in file_list:
        print(f"{pos}: {f}")
        pos += 1
    print("q: Exit program")
    ans = 0
    while ans != 'q':
        ans = input("Option: ")
        try:
            ans = int(ans)
            if ans > len(file_list):
                print(f"No such option: {ans}.")
            else:
                print(file_list[ans-1])
                nu_path = input("Enter new directory: ")
                if nu_path == "":
                    nu_path = "."
                csv_convert(path, file_list[ans-1], nu_path)
        except ValueError:
            pass
    return None


def mute():
    """
    `DataStruct` module method
    ---

    Description:
    ---
    Muting communicates about defaults in methods.
    """
    global communicate
    communicate = tuple("" for com in communicate)


def angular(x_arg, y_arg):
    """
    `DataStruct` module method
    ---

    Description:
    ---
    Angle coeff of two-way derivative

    Parameters:
    ---
    + `path` - specifies path to search for file to be converted
    Raises:
    ---
    None

    Returns:
    ---
    None

    If there is an error occuring it must be traced back to
    `DataStruct.listing` or `DataStruct.csv_convert` methods.
    """
    import math
    n_inte = y_arg
    y_arg.reverse()
    f_inte = y_arg

    d1 = deriv(x_arg, n_inte)
    d2 = deriv(x_arg, f_inte)
    d2.reverse

    a1 = [math.atan(-1/I) for I in d1]
    a2 = [math.atan(-1/I) for I in d2]

    ang = [abs(a2[n] + a1[n])/math.tau for n in range(len(d1))]
    return ang
