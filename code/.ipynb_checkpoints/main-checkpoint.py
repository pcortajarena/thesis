import os

images = []
def read_images_path(path, images_list):
    for x in os.listdir(path):
        path2 = os.path.join(path, x)
        if os.path.isdir(path2):
            print(path2)
            read_images_path(path2, images_list)
        elif os.path.isfile(path2):
            images.append(path2)