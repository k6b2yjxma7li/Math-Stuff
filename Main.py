import matplotlib.pyplot as plt
# import numpy as np
from DataStruct import DataStruct
import DataStruct as ds
# import Analysis


data = DataStruct("/home/rumcajs/Desktop/Inzynierka/Dane/raport_1/grafen_csv")


def geo_mean(data):
    cst = 1.0/len(data)
    mean = 1.0
    for n in range(len(data)):
        mean *= (data[n]**cst)
    return mean


data.get_files()
data.load_data()
data.real_part["name"] = data.header[0]
data.imag_part["name"] = data.header[1]
data.separation()
data.normalize(sum)

end = DataStruct.nearest(data.data[1]['Wave'],
                         data.data[0]['Wave'][-1])

beg = DataStruct.nearest(data.data[0]['Wave'],
                         data.data[1]['Wave'][0])

inte = ()
inte += (data.data[0]['Intensity'][beg:],)
inte += (data.data[1]['Intensity'][:end+1],)
inte += (data.data[2]['Intensity'][beg:],)

wave = ()
wave += (data.data[0]['Wave'][beg:],)
wave += (data.data[1]['Wave'][:end+1],)
wave += (data.data[2]['Wave'][beg:],)

dI = ()
dI += (DataStruct.deriv(wave[0], inte[0]),)
dI += (DataStruct.deriv(wave[1], inte[1]),)
dI += (DataStruct.deriv(wave[2], inte[2]),)

plt.figure(1)
for n in range(len(wave)):
    plt.plot(wave[n], inte[n], label="Raman spectrum "+str(n+1))
# plt.plot([wave[0][0], wave[0][-1]], [np.mean(inte[0]), np.mean(inte[0])])
plt.legend()

plt.figure(2)
plt.plot(wave[0], DataStruct.integral(wave[0], inte[0]), ".", markersize=1)


plt.show()

file_test = ds.listing("/home/rumcajs/Desktop/Inzynierka/")
# ds.csv_convert("/home/rumcajs/Desktop/Inzynierka/Dane/raport_1/grafen")
