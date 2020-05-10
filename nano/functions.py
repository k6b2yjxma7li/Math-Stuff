# The nano module

"""
`nano`'s `functions`
===
"""
import numpy as np
import os


def converter(filepath,
              action=lambda line: "\t".join([w for w in line.split("\t")
                                             if w]),
              file_ext=None,
              tmp_purge=False):
    """
    `nano`.`converter`
    ===
    Performs `action` text transformation over lines of file specified as
    `filepath`.
    """
    # import os
    dir, filename = os.path.split(filepath)
    if file_ext is not None:
        filename = os.path.basename(filename) + "." + file_ext
        filepath = os.path.join(dir, filename)
    tmpname = os.path.basename(filename) + ".tmp"
    tmpfile = os.path.join(dir, tmpname)
    # _, _, tmps = next(os.walk(dir))
    # print(tmps)
    # os.remove()
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
    """
    `nano`.`pearson`
    ===
    Computes value of Pearson's R correlation coefficient
    between `u` and `v` with a given expected value estimator
    function `estimator`.

    Returns
    ---
    Pearson's R correlation coefficient.
    """
    def cov(p, q, estimator=estimator):
        """
        `nano`.`pearson`.`cov`
        ---
        Computes value of covariance for `p` and `q` using
        `estimator` function as expected value estimator.

        Returns
        ---
        Covariance of `p` and `q`.
        """
        p = np.array(p)
        q = np.array(q)
        res = estimator(p*q) - estimator(p)*estimator(q)
        return res

    def stddev(t, estimator=estimator):
        """
        `nano`.`pearson`.`stddev`
        ---
        Computes value of standard deviation for `t` using
        `estimator` function as expected value estimator.

        Returns
        ---
        Standard deviation of `t`.
        """
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
    """
    `nano`.`div`
    ===
    Double point derivative of a `function` function, with step
    `dx`. Second parameter has to be tuned to reach maximal
    precision of derivative.

    Returns
    ---
    Function object of derivative.
    """
    # if x is not None and dx is None:
    #     dx = x[]
    def DIV(x):
        return (function(x+dx)-function(x-dx))/(2*dx)
    return DIV


def d(u):
    """
    `nano`.`d`
    ===
    Two-point discrete derivative of `u`.

    Returns
    ---
    Array of values corresponding to discrete derivative
    of function's main argument `u`.
    """
    u = np.array(u)
    center = list((u[2:]-u[:-2])/2)
    left = [(u[1]-u[0])]
    right = [(u[-1]-u[-2])]
    return np.array(left + center + right)


def gauss_av(sgm, u, v):
    """
    `nano`.`gauss_av`
    ===
    Special type of average. Uses Gauss curve as a window over
    data, where amplitude is driven by position of peak. Average
    applies for all points, which transforms data. The resulting
    array is smoothened.

    Arguments
    ---
    + `sgm`: float -- window width parameter, standard deviation
    of normal distribution (window)
    + `u`: array -- x-data
    + `v`: array -- y-data
    """
    def G(s):
        def GAUSS(u): return 1/(s*(2*np.pi)**(0.5)) * np.exp(-(u**2)/(2*s**2))
        return GAUSS
    res = []
    gs = G(sgm)
    for n in range(len(u)):
        g = gs(u-u[n])
        res += [sum(g*v*d(u))/(sum(g*d(u)))]
    return np.array(res)


def smoothing(t=[], mod=1):
    """
    `nano`.`smoothing`
    ===
    Similar to `gauss_av`. Performs averaging over data using
    second row of Pascal's triangle as a mask:
    (t[n-1]+2*t[n]+t[n+1])/4 = result

    Arguments
    ---
    + `t`: array -- data set
    + `mod`: int -- number of iterations (effective width of
    window)

    Returns
    ---
    Smoothened data (array).
    """
    t = np.array(t)

    def single_smoothing(f):
        df = (f[:-2]+2*f[1:-1]+f[2:])/4
        left = [(f[0]+f[1])/2]
        right = [(f[-2]+f[-1])/2]
        return np.append(left, np.append(df, right))

    for n in range(mod):
        t = single_smoothing(t)
    setattr(smoothing, "smooth", single_smoothing)
    return t


def xpeak(x_data, y_data, top_value, min_level):
    """
    `nano`:`xpeak`
    ===
    Extract points of single peak from data set.

    For a negative peak (dip) `y_data` and `top_value` have to negated.
    Function does not provide out-of-the-box support for extracting negative
    peaks.

    Arguments
    ---
    + `x_data`: array -- x-data
    + `y_data`: array -- y-data
    + `top_position`: float -- index of peak
    + `min_level`: float -- minimal level, where peak has it's bottom

    Returns
    ---
    (`itr_left`, `itr_peak`, `itr_right`)
        Array containing left boundary, peak and right boundary index values.
    """
    itr_peak = next(nearest_val(y_data, top_value))
    # y_peak = y_data[itr_peak]
    x_peak = x_data[itr_peak]
    itr_left = list(x_data).index(x_peak)
    itr_right = list(x_data).index(x_peak)
    for n in range(itr_right, len(x_data), 1):
        if y_data[n] <= min_level:
            break
    itr_right = n
    for k in range(itr_left, -1, -1):
        if y_data[k] <= min_level:
            break
    itr_left = k
    return itr_left, itr_peak, itr_right


if __name__ == "__main__":
    print(__doc__)
