from tqdm import tqdm
from utils import read_images_path, transform_dog, clean_images

    
if __main__ == '__name__':
    
    images = []
    read_images_path('/Users/pati/Documents/Thesis/thesis/src/img',
                     images)
    
    total = len(images)
    pbar = tqdm(total=total)
    
    for im in images:
        transform_dog(im, 0.8)
        pbar.update(1)

    clean_images('/Users/pati/Documents/Thesis/thesis/src/img2')
