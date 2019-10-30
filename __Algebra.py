"""
`Algebra module`
---
Description:
---
This module contains some handy algebraic functions basing
on `cmath` module and several patent methods and classes,
which create a powerful, but not super fast tool.

Classes:
---
+ Operator - parent class for all types of operator etc.

Methods:
---

Constants:
---
"""


def alen(arg):
    try:
        return len(arg)
    except TypeError:
        return 0


class Operator:
    def __init__(self, variables=((), (), ()), **kwargs):
        if len(variables) != len(kwargs):
            raise ValueError(f"Order type: {variables} "
                             f"(length of {len(variables)}) is "
                             "invalid due its structure comparing to "
                             "kwargs length.")
        for key, value in kwargs.items():
            setattr(self, key, value)

        def attr_check(from_arg, to_arg):
            if type(from_arg) in (list, tuple):
                return to_arg + from_arg
            else:
                return to_arg + (from_arg, )
        if alen(variables) == 0:
            self.mid_args = variables
        self.left_args = variables[0]
        self.mid_args = variables[1]
        self.right_args = variables[2]

    def __get_arg(self, arg_type, arg_num=0):
        arg = getattr(self, arg_type)
        if type(arg) in (list, tuple):
            return arg[arg_num]
        else:
            if arg_num != 0:
                print(f"Argument: {arg_type} is not iterable.")
            return arg

    def get_left(self, arg_num=0):
        return self.__get_arg("left_args", arg_num)

    def get_right(self, arg_num=0):
        return self.__get_arg("right_args", arg_num)

    def get_mid(self, arg_num=0):
        return self.__get_arg("mid_args", arg_num)


class Plus(Operator):
    def __init__(self, a, b):
        self = Operator(("p", "q"), p=a, q=b)
        setattr(self, "operation", "p+q")
        return eval(self.operation)
