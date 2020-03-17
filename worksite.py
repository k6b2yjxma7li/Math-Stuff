import numpy as np
import matplotlib.pyplot as plt
from nano.nano import d, nearest_val, table, gauss_av, smoothing, xpeak
from scipy.optimize import leastsq
import os

path = "./.data/Praca_inzynierska/Badania/190523/grf/"
_, _, files = next(os.walk(path))
datafiles = []
for n in range(len(files)):
    if ".txt" == files[n][-4:]:
        datafiles += [files[n]]


def spectrum(function, arg_count):
    def SPCTRM(u, params):
        if len(params) % arg_count != 0:
            raise IndexError("Parameters vector's length does not match "
                                "the declared length.")
        # arg_count = function.func_code.co_argcount
        data = np.zeros(len(u))
        for n in range(0, len(params), arg_count):
            data += function(params[n:n+arg_count], u)
        return data
    return SPCTRM


def asym(x_data, y_data, x0):
    split_pt = next(nearest_val(x_data, x0))
    left_y = y_data[:split_pt+1]
    right_y = y_data[split_pt:len(left_y)+split_pt]
    left_y = left_y[-len(right_y):]
    left_int = sum(left_y*d(x_data[split_pt-len(left_y):split_pt]))
    right_int = sum(right_y*d(x_data[split_pt:split_pt+len(right_y)]))
    return right_int-left_int
    # return np.cumsum((right_y-left_y)*)


def main():
    file_path = os.path.join(path, datafiles[0])
    print(file_path)
    tbl = table().read_csv(open(file_path, "r"))

    x = np.array(tbl['#Wave'])[::-1]
    y = np.array(tbl['#Intensity'])[::-1]

    def f(V, u): return V[0]/(((u-V[2])*V[1])**2 + 1)

    def res(V): return y - raman(x, V)

    def nRes2(V): return -np.cumsum(np.cumsum(res(V) * d(x)) * d(x))

    # fig, ax = plt.subplots()
    # # yd = np.cumsum(d(y))
    # yd = y - y[0]
    # Yd2 = np.cumsum(np.cumsum(yd*d(x))*d(x))
    # tp1 = xpeak(x, yd, max(yd), 0)
    
    # ax.plot(x, yd, '.', ms=0.9)
    # ax.plot(x[tp1[0]+1:tp1[-1]+1], yd[tp1[0]+1:tp1[-1]+1], '.', color="red")

    fig, ax = plt.subplots()
    ax.grid(True)
    ax.plot(x, y, '.', ms=0.9)

    scaling_factor = 0.95
    raman = spectrum(f, 3)

    x0 = x[int(len(x)/2)]
    y0 = y[int(len(x)/2)]
    residual = y
    sol = []
    for n in range(5):
        v1 = [y0, 0.01, x0]
        v0 = np.array(list(sol) + v1)
        y0 = residual[residual == max(residual)][0]
        x0 = x[residual == max(residual)][0]
        sol, hessian = leastsq(res, v0)
        ax.plot(x, raman(x, sol), lw=0.9, color=[0.1, 0.1, 0.1, (n+2)/12])

        print("Input:", str(v0[:3]).replace("\n", " ")[:-1] +
              "]"*(1-len(v0[3:4])) +
              f" and {len(v0[3:])} more..."*len(v0[3:4]))

        print("Solution:", str(sol[:3]).replace("\n", " ")[:-1] +
              "]"*(1-len(sol[3:4])) +
              f" and {len(sol[3:])} more..."*len(sol[3:4]))

        mask = np.arange(0, len(sol), 1) % 3 == 1
        sol[mask] *= scaling_factor
        residual = res(sol)
        print()

    sol[mask] /= scaling_factor
    ax.plot(x, raman(x, sol), lw=0.9, color="red")
    

    #   Preparing to extraction of maximum
    nRes2V = nRes2(sol)
    axd = ax.twinx()
    #   Depiction of nRes2
    axd.plot(x, np.abs(nRes2V), lw=0.8, color='black')
    axd.plot(x[np.abs(nRes2V) == max(np.abs(nRes2V))],
             nRes2V[np.abs(nRes2V) == max(np.abs(nRes2V))],
             '.', color=[1, 0, 1, 1])
    #   Proper peak params extraction
    ix = xpeak(x, nRes2V, max(nRes2V), max(nRes2V)/2)
    #   Half-max width
    # hmw = abs(ix[0]-ix[-1])
    # x0 = list(nRes2V).index(min(nRes2V))
    # Amp = sum(f([1, 1/hmw, x[x0]], x)*res(sol)/sum(f([1, 1/hmw, x[x0]], x))*d(x))
    # print([Amp, 1/hmw, x[x0]])
    # sol, hess = leastsq(res, list(sol)+[Amp, 1/hmw, x0])
    # ax.plot(x, raman(x, sol), lw=1.5, color=[1, 0, 1, 1])
    # axd.plot(x, nRes2(sol), '--', lw=0.9, color='black')
    # ax.plot(x, res(sol), '.', ms=0.9, color='red')
    plt.show()


if __name__ == "__main__":
    main()
