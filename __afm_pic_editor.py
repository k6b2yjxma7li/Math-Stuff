""""
version: newver
"""
import matplotlib.pyplot as plt
import matplotlib_scalebar.scalebar as sb

plt.rc('text', usetex=True)
plt.rc('font', family='serif')

plt.figure()

image = plt.imread("/home/rumcajs/Desktop/Praca_inzynierska"
                   "/Badania/180719/grafiki/hbn_1_5.jpg").copy()
fig1 = plt.figure(1)
new_img = []
for n in range(len(image)-20):
    line = [image[n][k] for k in range(len(image[0])) if (120 < k < 860) or (k > 960)]
    new_img.append(line)
image = new_img
plt.box(False)
plt.imshow(image)
scale = sb.ScaleBar(0.02, 'um', location="upper left")
plt.xlabel("Columns")
plt.ylabel("Rows")
plt.gca().add_artist(scale)
plt.axis('off')
plt.savefig("./hbn1-5.png", format="png", dpi=500, bbox_inches="tight")
plt.show()
