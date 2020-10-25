"""
`nano`'s `table` class
===
"""
import re
import numpy as np


class table:
    """
    `table`
    ===
    Class created to read, store, access and export tables like CSV.
    """
    def __init__(self, plain_dict={}, delim=","):
        """
        `table` constructor (from plain `dict`)
        """
        self.__dict_ = plain_dict
        self.__delim_ = delim
        self._err_ = []

    def __str__(self):
        result = ""
        for k, v in self.__dict_.items():
            if hasattr(v, '__len__'):
                if len(v) > 4:
                    result += (f"{k}: {str(v[:3])[:-1]}" +
                               f" and {len(v)-3} more...\n")
                else:
                    result += (f"{k}: {str(v)}\n")
            else:
                result += f"{k}: {v}\n"
        if result:
            return ("table object:" + ("\n"+result).replace("\n", "\n\t"))[:-2]
        else:
            return "table object: {}"

    def __repr__(self):
        return "#t:"+str(self.__dict_)

    def __add__(self, tbl):
        if type(tbl) is dict:
            tbl = table(tbl)
        elif type(tbl) is not table:
            raise TypeError(f"Addition not allowed between table"
                            f" and {str(type(tbl))}.")
        __sdict_cp = self.__dict_.copy()
        __edict_cp = tbl.__dict_.copy()
        __sdict_cp.update(__edict_cp)
        return table(__sdict_cp, self.__delim_)

    def __getitem__(self, key):
        result = {}
        if type(key) is slice:
            for k, v in self.__dict_.items():
                result.update({k: v[key]})
        elif type(key) is np.ndarray:
            for k, v in self.__dict_.items():
                result.update({k: np.array(v)[key]})
        elif key in self.__dict_.keys():
            result = self.__dict_[key]
        elif type(key) is int:
            for k, v in self.__dict_.items():
                result.update({k: v[key]})
        elif type(key) == str:
            if key[0] == '#':
                try:
                    key = int(key[1:])
                    keys = list(self.keys())
                    result = self[keys[key]]
                except ValueError:
                    pass
                except IndexError:
                    pass
        return result

    def __setitem__(self, key, value):
        self.__dict_[key] = value
        return self

    @property
    def __err__(self):
        """
        `table`'s `__err__`
        ---
        Error log for data access.
        """
        if len(self._err_) == 0:
            return "No errors encountered"
        else:
            return f"{len(self._err_)} errors encountered:\n{self._err_}"

    def keys(self):
        return self.__dict_.keys()

    def items(self):
        return self.__dict_.items()

    def pop(self, key):
        return self.__dict_.pop(key)

    def retype(self, cell_key, dtype):
        self[cell_key] = dtype(self[cell_key])

    def sort(self, key=lambda x: x, reverse=False):
        new_keys = sorted(list(self.__dict_.keys()), key=key, reverse=False)
        new_vals = [self.__dict_[key] for key in new_keys]
        new_tbl = table(dict(zip(new_keys, new_vals)), delim=self.__delim_)
        new_tbl._err_ = self._err_
        return new_tbl

    def read_csv(self, file_stream, delim=None, is_header=True, verb=False):
        """
        CSV reader storing data in self `nano`.`table` .
        """
        if delim is None:
            delim = self.__delim_
        # import csv
        _err_ = []
        if re.search(r"\(.*\)", delim) is not None and verb:
            print(f"Warning: Delimiter regex `{delim}` contains group match `()`; "
                   "this pose a risk of unintentional split results.")
        # main separating function is re's split
        if is_header:
            header = re.split(delim, next(file_stream)[:-1])
            fst_line = re.split(delim, next(file_stream)[:-1])
        else:
            fst_line = re.split(delim, next(file_stream)[:-1])
            header = list(map(lambda q: "$"+str(q), range(len(fst_line))))
        fst_line = list(map(lambda q: [q], fst_line))
        self.__dict_ = dict(zip(header, fst_line))
        # new_self = table(dict(zip(header, fst_line)))
        for line in file_stream:
            data_line = re.split(delim, line[:-1])
            # for n, key in enumerate(new_self.keys()):
            for n, key in enumerate(self.__dict_.keys()):
                try:
                    self.__dict_[key] += [data_line[n]]
                    # new_self[key] += [data_line[n]]
                except IndexError as e:
                    self.__dict_[key] += [None]
                    # new_self[key] += [None]
                    _err_ += [e]
        # for key, val in new_self.items():
        for key, val in self.__dict_.items():
            for n, v in enumerate(val):
                try:
                    if '.' in str(v):
                        self.__dict_[key][n] = float(v)
                        # new_self[key][n] = float(v)
                    else:
                        self.__dict_[key][n] = int(v)
                        # new_self[key][n] = int(v)
                except (ValueError, TypeError) as e:
                    self.__dict_[key][n] = v
                    # self[key][n] = v
                    _err_ += [e]
        file_stream.close()
        self._err_ = _err_
        # setattr(table.read_csv, 'err', _err_)
        # return new_self
        return self

    def export_csv(self, file_stream, delim=None, is_header=True,
                   header=None):
        if delim is None:
            delim = self.__delim_
        col_lens = list(map(lambda c: len(c), list(self.__dict_.values())))
        try:
            line = ""
            if is_header:
                if header is None:
                    header = list(self.__dict_.keys())
                if len(header) != len(list(self.__dict_.keys())):
                    raise IndexError("Inserted header does not properly map to"
                                     f" table's columns: {header} <>"
                                     f" {list(self.__dict_.keys())}")
                for nr, key in enumerate(header):
                    line += str(header[nr])+delim
                print(line[:-1], file=file_stream)
            for n in range(max(col_lens)):
                line = ""
                for k in self.__dict_.keys():
                    try:
                        line += str(self[k][n])+delim
                    except IndexError:
                        line += delim
                print(line[:-1], file=file_stream)
        except Exception as e:
            # print(re.sub(r"(\<class '|'\>)", "", str(type(e)))+": "+str(e))
            raise e
        file_stream.close()
