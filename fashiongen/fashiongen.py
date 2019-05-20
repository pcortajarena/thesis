import h5py
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from tqdm import tqdm

filename = 'fashiongen/fashion_val.h5'
f = h5py.File(filename, 'r')

keys = list(f.keys())

# save images from file
i = 1
total = len(f['input_image'])
p = tqdm(total=total)

for image in f['input_image']:
    im = Image.fromarray(image)
    im.save('images/fashiongen/{}.jpg'.format(i))
    i += 1
    p.update(1)

# process descriptions
data = pd.DataFrame()
data['name'] = [desc[0].decode('latin-1') for desc in list(f['input_name'])]

# save descriptions in csv from pandas
data.to_csv('fashiongen/descriptions.csv', index=False)

# apply tf-idf to documents
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(list(data['name']))