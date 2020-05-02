import re

operators = ["+", "-", "*", "/"]


equation = "x**2"


def deriv(equation, var_name):
    power_of_re = r"(?<=\s)[\s\S]+?\*\*[\s\S]+?(?=\s)"
    base_re = r"[\s\S]+?(?=\*\*)"
    power_re = r"(?<=\*\*)[\s\S]"

    full_match = equation[slice(*re.search(power_of_re, " "+equation+" "))]
    print(full_match)


    

print(deriv("x**2", "x"))