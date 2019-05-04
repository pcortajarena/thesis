import h5py
from PIL import Image

filename = 'fashion_val.h5'
f = h5py.File(filename, 'r')

keys = list(f.keys())

# save images from file
i = 1
for image in f['input_image']:
    im = Image.fromarray(image)
    im.save('images/fashiongen2/{}.jpg'.format(i))
    i += 1
