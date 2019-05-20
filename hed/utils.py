import os
from PIL import Image
from shutil import copyfile

def read_images_path(path, images_list):
    for x in os.listdir(path):
        path2 = os.path.join(path, x)
        if os.path.isdir(path2):
            read_images_path(path2, images_list)
        elif os.path.isfile(path2) and path2.endswith('.jpg') and 'combine' not in path2:
            images_list.append(path2)

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

def divide_train_test(path):
    images = os.listdir(path)
    total = len(images)
    train = total * 2 // 3
    i = 0
    for im in images:
        if i < train:
            copyfile('images/fashiongen/{}'.format(im), 'images/train/{}'.format(im))
            i += 1
        else:
            copyfile('images/fashiongen/{}'.format(im), 'images/test/{}'.format(im))
            i += 1
    
    