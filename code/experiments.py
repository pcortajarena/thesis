from keras.models import load_model
from dataloader import DataLoader
import matplotlib.pyplot as plt

# import models
generator_text = load_model('models/generator_text.h5')
generator_simple = load_model('models/generator_simple.h5')

# create data loader and load data
data = DataLoader('fashiongen', 200)
batch = 50
imgs_A, imgs_B, text_desc = data.load_data(batch_size=batch, is_testing=True)

# predict values for simple and text gan
fake_simple = generator_simple.predict([imgs_B])
fake_text = generator_text.predict([imgs_B, text_desc])

# save images
for i in range(batch):
    plt.imshow(fake_simple[i])
    plt.axis('off')
    plt.savefig("images/experiments/{}_{}.png".format(i, 'simple'))
    plt.close()

    plt.imshow(fake_text[i])
    plt.axis('off')
    plt.savefig("images/experiments/{}_{}.png".format(i, 'text'))
    plt.close()

    plt.imshow(imgs_A[i])
    plt.axis('off')
    plt.savefig("images/experiments/{}_{}.png".format(i, 'original'))
    plt.close()