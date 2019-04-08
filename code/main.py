import os
import matplotlib.pyplot as plt
from skimage import color, filters, img_as_float, img_as_ubyte
from skimage.feature import blob_dog
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_images_path(path, images_list):
    for x in os.listdir(path):
        path2 = os.path.join(path, x)
        if os.path.isdir(path2):
            read_images_path(path2, images_list)
        elif os.path.isfile(path2) and path2.endswith('.jpg'):
            images.append(path2)
            

def transform_dog(image_path, sigma):
    img = plt.imread(image_path)
    img = img_as_float(img)
    img_gray = color.rgb2gray(img)
    
    k = 1.8
    s1 = filters.gaussian(img_gray ,k*sigma)
    s2 = filters.gaussian(img_gray ,sigma)
    dog = (s1 - s2)
    dog = 255 - img_as_ubyte(dog)
    
    path2 = image_path + '_dog.jpg'
    plt.imsave(path2, dog, cmap='gray')
    
if __main__ == '__name__':
    
    images = []
    read_images_path('/Users/pati/Documents/Thesis/thesis/src/img',
                     images)
    
    total = len(images)
    pbar = tqdm(total=total)
    
    for im in images:
        transform_dog(im, 0.8)
        pbar.update(1)