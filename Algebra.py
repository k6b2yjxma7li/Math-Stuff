import numpy as np


class Matrix:
    def __init__(self, matrix=[], func=str):
        self._matrix = ()
        self._func = func
        for k in range(len(matrix)):
            self._matrix += ((),)
            for l in range(len(matrix[k])):
                self._matrix[k] += (func(matrix[k][l]),)

    def __str__(self):
        str_self = ""
        print(f"dim:{len(self._matrix), len(self._matrix[0])}")
        for row in self._matrix:
            str_self += row.__str__() + ",\n"
        return str_self[:-2]+";\n"

    def transpose(self):
        """
        Transposition of matrix.
        """
        # print(f"dim:{len(self.matrix), len(self.matrix[0])}")
        result = ()
        for k in range(len(self._matrix)):
            result += ((),)
            for l in range(len(self._matrix[k])):
                result[k] += (self._matrix[l][k],)
        return self.__class__(result)

    def get(self):
        return self._matrix

    def get_func(self):
        return self._func

    def get_diag(self):
        result = ()
        for n in range(len(self._matrix)):
            result += (self._matrix[n][n],)
        return result

    def mult(self, mat=None):
        if mat is None:
            return multiplication(self, self)
        else:
            return multiplication(self, mat)


class Vector(Matrix):
    def __init__(self, matrix=[], func=str):
        self._matrix = ()
        self._func = func
        for n in range(len(matrix)):
            self._matrix += ((func(matrix[n])), )

    def __str__(self):
        str_self = ""
        print(f"dim:{len(self._matrix), len(self._matrix[0])}")
        for n in range(len(self._matrix)):
            str_self += self._matrix[n].__str__()+",\n"
        return str_self[:-2]+";\n"

    def norm_sq(self):
        return Equation(multiplication(self.transpose(), self).get()[0][0])

    def to_matrix(self):
        result = ()
        for k in range(len(self._matrix)):
            result += ((),)
            for l in range(len(self._matrix)):
                if k == l:
                    result[k] += (self._matrix[k][0],)
                else:
                    result[k] += ("0",)
        return Matrix(result)

    def transpose(self):
        result = ((),)
        for elmnt in self._matrix:
            result[0] += ((elmnt[0],),)
        full_res = Matrix(result)
        full_res.__class__ = Vector
        return full_res


def multiplication(M1=Matrix(), M2=Matrix(), func=str):
    def wrap(val):
        if len(val) > 1:
            val = "(" + val + ")"
        return val
    M1 = M1.get()
    M2 = M2.get()
    result = ()
    for k in range(len(M1)):
        result += ((),)
        for l in range(len(M2[0])):
            element = ""
            for m in range(len(M1[0])):
                ele1 = M1[k][m]
                ele2 = M2[m][l]
                if ele1 == "1" and ele2 == "1":
                    element += "1 + "
                elif ele1 != "0" and ele2 != "0":
                    ele1 = wrap(ele1)
                    ele2 = wrap(ele2)
                    element += "(" + ele1 + " * " + ele2 + ") + "
                element = element.replace(" * 1)", ")")
                element = element.replace("(1 * ", "(")
            if len(element) == 0:
                element = "0 + "
            result[k] += (func(element[:-3]),)
    return Matrix(result)


def identity(obj):
    result = ()
    if obj.__class__ == Matrix:
        for n in range(len(obj.get())):
            result += ((),)
            for k in range(len(obj.get()[n])):
                if n == k:
                    result[n] += (obj.get()[n][n],)
                else:
                    result[n] += ("0",)
    if obj.__class__ == Vector:
        for n in range(len(obj.get())):
            result += ((),)
            for k in range(len(obj.get())):
                if n == k:
                    result[n] += (obj.get()[n],)
                else:
                    result[n] += ("0",)
    return Matrix(result)


class Equation:
    def __init__(self, strg="", func=str):
        self.strg = func(strg)

    def __str__(self):
        return "Equation:\n" + self.get().__str__()

    def get(self):
        return self.strg

    def eval_me(self):
        return eval(self.get())

    def mult(self, equ):
        if equ.__class__ is not Equation:
            raise TypeError(f"Value '{equ}' is not a valid equation.")
        return Equation("(" + self.get() + ") * (" + equ.get() + ")")

    def pow(self, equ):
        if equ.__class__ is not Equation:
            raise TypeError(f"Value '{equ}' is not a valid equation.")
        return Equation("(" + self.get() + ") ** (" + equ.get() + ")")

    @staticmethod
    def sin(equ):
        if equ.__class__ is not Equation:
            raise TypeError(f"Value '{equ}' is not a valid equation.")
        return Equation("np.sin(" + equ.get() + ")")
