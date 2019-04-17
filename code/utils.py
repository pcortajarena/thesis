import os
from skimage import color, filters, img_as_float, img_as_ubyte
from skimage.feature import blob_dog
import matplotlib.pyplot as plt
from PIL import Image

def read_images_path(path, images_list):
    for x in os.listdir(path):
        path2 = os.path.join(path, x)
        if os.path.isdir(path2):
            read_images_path(path2, images_list)
        elif os.path.isfile(path2) and path2.endswith('.jpg'):
            images_list.append(path2)

def transform_dog(image_path, sigma, dest_path):
    img = plt.imread(image_path)
    img = img_as_float(img)
    img_gray = color.rgb2gray(img)
    
    k = 1.8
    s1 = filters.gaussian(img_gray ,k*sigma)
    s2 = filters.gaussian(img_gray ,sigma)
    dog = (s1 - s2)
    dog = 255 - img_as_ubyte(dog)
    
    path2 = image_path[:-4] + '_dog.jpg'
    plt.imsave(path2, dog, cmap='gray')
    
    combine_pictures(image_path, path2, dest_path)
    
def combine_pictures(path_original, path_dog, dest_path):
    img = Image.open(path_original)
    img2 = Image.open(path_dog)
    images = [img, img2]
    widths, heights = zip(*(i.size for i in images))
    
    total_width = sum(widths)
    max_height = max(heights)
    
    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]
    
    new_im.save(dest_path)
    
def clean_images(path):
    
    for x in os.listdir(path):
        path2 = os.path.join(path, x)
        if os.path.isdir(path2):
            clean_images(path2)
        elif os.path.isfile(path2) and not path2.endswith('combine.jpg'):
            os.remove(path2)