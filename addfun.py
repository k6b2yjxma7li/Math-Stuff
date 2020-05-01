import numpy as np
from nano.functions import d, smoothing


def chisq(data, model):
    """`chisq`
    ---
    Chi-squared distribution value

    Parameters
    ----------
    data : `array`
        Observed values\\
    model : `array`
        Expected values

    Returns
    -------
    `float`
        Value of chi-squared distribution
    """
    data = np.array(data)
    model = np.array(model)
    return sum(((data-model)**2)/model)


def spectrum(function, arg_count):
    """`spectrum`
    ---
    Spectral shape made from multiple curves

    Parameters
    ----------
    function : `function obj`
        Peak/curve function to define spectrum model\\
    arg_count : `int`
        Number of parameters per curve

    Returns
    -------
    `function obj`
        Object of a function that represents spectral shape
    """
    def SPCTRM(params, u):
        """`SPCTRM`
        ---
        `spectrum` result function

        Parameters
        ----------
        params : `array`
            Vector-like argument containing all peak parameters\\
        u : `array`
            Abscissa argument of spectral curve (x data)

        Returns
        -------
        `array`
            Values calculated for spectral parameters and functions

        Raises
        ------
        `IndexError`:
            when length of `params` vector does not match `spectrum`'s
            `arg_count`.
        """
        if len(params) % arg_count != 0:
            raise IndexError("Parameters vector's length does not match "
                             "the declared length.")
        data = np.zeros(len(u))
        for n in range(0, len(params), arg_count):
            data += function(params[n:n+arg_count], u)
        return data
    return SPCTRM


def spectrum2(function, arg_count=3):
    """`spectrum2`
    ---
    Spectral shape made from multiple curves

    Difference between this function and `spectrum` is
    order by which `params` are read; first N values are
    first arguments of a `function`, following N values are
    second arguments, etc.

    Parameters
    ----------
    function : `function obj`
        Peak/curve function to define spectrum model\\
    arg_count : `int`, optional
        Number of parameters per curve, by default 3

    Returns
    -------
    `function obj`
        Object of a function that represents spectral shape
    """
    def SPCTR2(params, u):
        """`SPCTR2`
        ---
        `spectrum2` result function

        Parameters
        ----------
        params : `array`
            Vector-like argument containing all peak parameters\\
        u : `array`
            Abscissa argument of spectral curve (x data)

        Returns
        -------
        `array`
            Values calculated for spectral parameters and functions

        Raises
        ------
        `IndexError`:
            when length of `params` vector does not match `spectrum2`'s
            `arg_count`.
        """
        data = np.zeros(len(u))
        u = np.array(u)
        func_count = int(len(params)/arg_count)
        if len(params)//func_count != len(params)/func_count:
            raise IndexError("Parameters vector's length does not match "
                             "the declared number of functions.")
        ix = np.arange(len(params))  # keys to perform rule of choice
        for k in range(func_count):
            # rule of choice: ix % func_count == k
            data += function(params[ix % func_count == k], u)
        return data
    return SPCTR2


def residual(func, x, y, penalty=None):
    """`residual`
    ---
    Difference between points of model (`func(x)`) and observed
    values (`y`) -- fit residuals

    Parameters
    ----------
    func : `function obj`
        Model function; must take two arguments: parameter vector and
        x-data
    x : `array`
        Abscissa\\
    y : `array`
        Observed values\\
    penalty : `array`
        Deviation coeff; if left as `None` is 1 over all pairs
        of x's and y's, default None


    Returns
    -------
    `function obj`
        Residual function, depending on fit parameters vector
    """
    if penalty is None:
        penalty = np.linspace(1, 1, len(y))

    def RES(V):
        """`RES`
        ---
        Result function of `residual`

        Parameters
        ----------
        V : `array`
            Parameters vector for fit function

        Returns
        -------
        `array`
            Difference between y-data and fitted points
        """
        ye = func(V, x)
        return (y-ye)/penalty
    return RES


def lorentz(V, t):
    t = np.array(t)
    return abs(V[0])/(1+((t-V[2])/V[1])**2)


def integr(y_arg, x_arg):
    x_arg = np.array(x_arg)
    y_arg = np.array(y_arg)
    i = np.linspace(2, 2, len(y_arg))
    i[0] = 1
    i[-1] = 1
    return np.cumsum(y_arg*d(x_arg)*i)


def smoothen(u):
    """Single-iterational `smoothing` function"""
    left = [(u[0] + u[1])/2]
    center = list((u[:-2] + 2*u[1:-1] + u[2:])/4)
    right = [(u[-2] + u[-1])/2]
    return np.array(left+center+right)


def D(u):
    """Reversing `smoothen`

        D(u) + smoothen(u) = u
    """
    u = np.array(u)
    left = [(u[0]-u[1])/2]
    right = [(u[-1] - u[-2])/2]
    center = list((-u[:-2] + 2*u[1:-1] - u[2:])/4)
    return np.array(left + center + right)


def gdev(arr, mod=1):
    if mod < 1:
        mod = 1
    return (smoothing(arr**2, mod) - smoothing(arr, mod)**2)**0.5
