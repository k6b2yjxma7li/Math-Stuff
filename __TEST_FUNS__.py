
def vectorize(vector, length):
    return [vector[n:n+length] for n in range(0, len(vector), length)]


def d_func(func, h):
    def div(x):
        return (func(x+h)-func(x-h))/(2*h)
    return div


def d2_func(func, h):
    def div(x):
        return (func(x+h)-2*func(x)+func(x-h))/(h**2)
    return div


def tangent(x, function):
    dfunc = d_func(function, 1e-9)
    a = dfunc(x)  # a-value of line
    b = function(x)-a*x    # b-value of line
    return (a, b)


def fano(q, G, E0, E):
    return ((q*G/2 + E - E0)**2)/((G/2)**2 + (E-E0)**2)


def div(dim, func, h):
    def DIV(point):
        p1 = point.copy()
        p2 = point.copy()
        p1[dim] -= h
        p2[dim] += h
        return (func(*p2)-func(*p1))/(2*h)
    return DIV


def div2(dim, func, h):
    def DIV2(point):
        p1 = point.copy()
        p2 = point.copy()
        p1[dim] -= h
        p2[dim] += h
        return (func(*p2)-2*func(*point)+func(*p1))/(h**2)
    return DIV2


def quadratic_peak(a, b, c, x0):
    """
    a(x-x0)**2+y0 =
    = a*x**2 - 2*a*x0*x + a*x0**2 + y0
    =>a*x**2 + b*x      + c
    -2*a*x0 = b
    c = a*x0**2 + y0
    x0 = -b/(2*a)
    y0 = c - (b**2)/(4*a)
    """
    if a != 0:
        x0 = -b/(2*a)
    return x0, a*x0**2 + b*x0 + c


def gradient(func, h):
    def GRAD(point):
        grd = []
        for dim in range(len(point)):
            p1 = point.copy()
            p2 = point.copy()
            p1[dim] -= h
            p2[dim] += h
            grd.append((func(*p2) - func(*p1))/(2*h))
        return grd
    return GRAD


def norm(vector):
    return sum(vector**2)
