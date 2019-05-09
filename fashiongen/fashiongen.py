import h5py
from PIL import Image
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

filename = 'fashion_val.h5'
f = h5py.File(filename, 'r')

keys = list(f.keys())

# save images from file
i = 1
for image in f['input_image']:
    im = Image.fromarray(image)
    im.save('images/fashiongen2/{}.jpg'.format(i))
    i += 1

# process descriptions
data = pd.DataFrame()
data['name'] = [desc[0].decode('latin-1') for desc in list(f['input_name'])]

# save descriptions in csv from pandas
data.to_csv('descriptions.csv', index=False)

# apply tf-idf to documents
vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(list(data['name']))