import DataStruct as ds

ds.mute()


def floor(value):
    return int(value)


def ceil(value):
    return int(value)+1


data = ds.DataStruct("D:/Praca_inzynierska/Badania/190523/grf")

# data = ds.DataStruct("./")

data.get_files()
data.load_data()
data.real_part["name"] = data.header[0]
data.imag_part["name"] = data.header[1]
data.separation("Wave", "Intensity")
data.normalize(sum)

end = ds.nearest(data.data[1]['Wave'],
                 data.data[0]['Wave'][-1])

beg = ds.nearest(data.data[0]['Wave'],
                 data.data[1]['Wave'][0])

inte = ()
inte += (data.data[0]['Intensity'][beg:],)
inte += (data.data[1]['Intensity'][:end+1],)
inte += (data.data[2]['Intensity'][beg:],)

wave = ()
wave += (data.data[0]['Wave'][beg:],)
wave += (data.data[1]['Wave'][:end+1],)
wave += (data.data[2]['Wave'][beg:],)

# inte = ()
# inte += (data.data[0]['#Intensity'],)

# wave = ()
# wave += (data.data[0]['#Wave'],)
