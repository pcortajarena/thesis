from tqdm import tqdm
from utils import read_images_path, transform_dog, clean_images

    
if __main__ == '__name__':
    
    images = []
    read_images_path('/Users/pati/Documents/Thesis/thesis/src/img3', images)
    
    total = len(images)
    pbar = tqdm(total=total)
    
    i = 1
    for im in images:
        if i < 175000:
            transform_dog(im, 0.8, 
                          '/Users/pati/Documents/Thesis/thesis/src/train/{}.jpg'.format(i))
            i += 1
        elif i >= 175000 and i < 233000:
            transform_dog(im, 0.8, 
                          '/Users/pati/Documents/Thesis/thesis/src/val/{}.jpg'.format(i))
            i += 1
        else:
            transform_dog(im, 0.8, 
                          '/Users/pati/Documents/Thesis/thesis/src/test/{}.jpg'.format(i))
            i += 1
        pbar.update(1)

    clean_images('/Users/pati/Documents/Thesis/thesis/src/img2')
