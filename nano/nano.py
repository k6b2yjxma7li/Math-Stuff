# The nano module

"""
The nano module
===

Collection of useful elements, data structures, functions, etc.
"""
import numpy as np
import csv
import os


class table(dict):
    """
    `nano`.`table`
    ===
    Dict-based table with column/row access to data.

    Data can be accessed row-wise (numerical) or column-wise.
    """
    def __init__(self, plain_dict):
        """
        `nano`.`table` constructor (from plain `dict`)
        """
        self.__dict__ = plain_dict

    def read_csv(self, file_stream, delim='\t'):
        """
        CSV reader storing data in self `nano`.`table` .
        """
        # import csv
        self.__fstream__ = file_stream
        self.__delim__ = delim
        csvdict = csv.DictReader(self.__fstream__, delimiter=delim)
        self.__dict__ = {}
        _csv_ = dict(next(csvdict))
        for k, v in _csv_.items():
            try:
                self.__dict__[k] = [float(v)]
            except ValueError:
                self.__dict__[k] = [v]
        for line in csvdict:
            for k, v in line.items():
                try:
                    self.__dict__[k] += [float(v)]
                except ValueError:
                    self.__dict__[k] += [v]

    def __str__(self):
        result = ""
        for k, v in self.__dict__.items():
            if hasattr(v, '__len__'):
                if len(v) > 4:
                    result += (f"\t{k}: {str(v[:3])[:-1]}" +
                               f" and {len(v)-3} more...\n")
            else:
                result += f"\t{k}: {v},\n"
        return "table object:\n" + result

    def __getitem__(self, key):
        result = {}
        if type(key) is slice:
            for k, v in self.__dict__.items():
                result.update({k: v[key]})
        elif key in self.__dict__.keys():
            result = self.__dict__[key]
        elif type(key) is int:
            for k, v in self.__dict__.items():
                result.update({k: v[key]})
        return result

    def __add__(self, tbl):
        sf = self.copy()
        return sf.update(tbl)


def converter(filepath,
              action=lambda line: "\t".join([w for w in line.split("\t")
                                             if w])):
    """
    `nano`.`converter`
    ===
    Performs `action` text transformation over lines of file specified as
    `filepath`.
    """
    # import os
    dir, filename = os.path.split(filepath)
    tmpname = os.path.basename(filename) + ".tmp"
    tmpfile = os.path.join(dir, tmpname)
    if os.path.isfile(filepath):
        try:
            os.rename(filepath, tmpfile)
            with open(tmpfile, 'r') as infile:
                print(tmpfile, "--> ", end="")
                with open(filepath, 'w') as outfile:
                    print(filepath)
                    while True:
                        try:
                            line = next(infile)
                            print(action(line), file=outfile, end='')
                        except (EOFError, StopIteration):
                            break
        except FileExistsError:
            print("File not converted: tmp version already exists.",
                  "Haven't you converted the file already?")
            pass
    else:
        raise IsADirectoryError(f"File {filepath} does not exist or is a" +
                                " directory!")


def nearest_val(iterable, value):
    """
    `nano`.`nearest_val`
    ===
    Sorts `iterable` depending on proximity to `value` and returns
    set of indexes which correspond to order of proximity.

    Returns
    ---
    Iterator over indexes of `iterable`.
    """
    # import numpy as np
    iterable = np.array(iterable)
    abs_iter = list(np.abs(iterable - value))
    # index = range(len(abs_iter))
    s = sorted(abs_iter)
    return (abs_iter.index(d) for d in s)


def pearson(u, v, estimator=np.mean):
    def cov(p, q, estimator=estimator):
        p = np.array(p)
        q = np.array(q)
        res = estimator(p*q) - estimator(p)*estimator(q)
        return res 

    def stddev(t, estimator=estimator):
        t = np.array(t)
        res = (estimator(t**2) - estimator(t)**2)**0.5
        return res

    setattr(pearson, 'estimator_f', estimator)
    setattr(pearson, 'covariance_f', cov)
    setattr(pearson, 'stddev_f', stddev)
    setattr(pearson, 'estimator_v', (estimator(u), estimator(v)))
    setattr(pearson, 'covariance_v', cov(u, v))
    setattr(pearson, 'stddev_v', (stddev(u), stddev(v)))
    return cov(u, v, estimator)/(stddev(u, estimator)*stddev(v, estimator))


def div(function, dx):
    # if x is not None and dx is None:
    #     dx = x[]
    def DIV(x):
        return (function(x+dx)-function(x-dx))/(2*dx)
    return DIV


if __name__ == "__main__":
    print(__doc__)
