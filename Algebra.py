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

Methods:
---

Constants:
---
"""

import cmath

class Operator:
    def __init__(self, operation, arg1, arg2=None):
        self.operation = operation
        self.arg1 = arg1
        if arg2 is not None:
            self.arg2 = arg2

    def __str__(self):
        if self.arg2 is not None:
            return f"{self.arg1} {self.operation} {self.arg2}"
        else:
            return f"{self.operation} {self.arg1}"
