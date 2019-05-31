from __future__ import print_function, division
import scipy

from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
import datetime
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from dataloader import DataLoader

class Pix2Pix():
    def __init__(self, simple_gan=False, text_gan=True):

        # Initilize variables
        self.simple_gan = simple_gan
        self.text_gan = text_gan

        # Input shape
        self.img_rows = 256
        self.img_cols = 256
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        self.max_features = 512
        self.text_shape = (1, 1, self.max_features)

        # Configure data loader
        self.dataset_name = 'fashiongen'
        self.data_loader = DataLoader(dataset_name=self.dataset_name, features=self.max_features,
                                      img_res=(self.img_rows, self.img_cols))


        # Calculate output shape of D (PatchGAN)
        patch = int(self.img_rows / 2**4)
        self.disc_patch = (patch, patch, 1)

        # Number of filters in the first layer of G and D
        self.gf = 64
        self.df = 64

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator (with or without text input)
        if self.simple_gan:
            self.discriminator_simple = self.build_discriminator(text_gan=False)
            self.discriminator_simple.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        if self.text_gan:
            self.discriminator_text = self.build_discriminator(text_gan=True)
            self.discriminator_text.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        #-------------------------
        # Construct Computational
        #   Graph of Generator
        #-------------------------

        # Input images and their conditioning images
            img_A = Input(shape=self.img_shape)
            img_B = Input(shape=self.img_shape)
            text_input = Input(shape=self.text_shape)
            text_input_d = Reshape(target_shape=(self.max_features,))(text_input)

        # Build the generator (with or without text input)
        if self.simple_gan:
            self.generator_simple = self.build_generator(text_gan=False)

            # By conditioning on B generate a fake version of A
            fake_A = self.generator_simple([img_B, text_input])

            # For the combined model we will only train the generator
            self.discriminator_simple.trainable = False

            # Discriminators determines validity of translated images / condition pairs
            valid = self.discriminator_simple([fake_A, img_B, text_input_d])

            self.combined_simple = Model(inputs=[img_A, img_B, text_input], outputs=[valid, fake_A])
            self.combined_simple.compile(loss=['mse', 'mae'],
                                loss_weights=[1, 100],
                                optimizer=optimizer)
        
        if self.text_gan:
            self.generator_text = self.build_generator(text_gan=True)

            # By conditioning on B generate a fake version of A
            fake_A = self.generator_text([img_B, text_input])

            # For the combined model we will only train the generator
            self.discriminator_text.trainable = False

            # Discriminators determines validity of translated images / condition pairs
            valid = self.discriminator_text([fake_A, img_B, text_input_d])

            self.combined_text = Model(inputs=[img_A, img_B, text_input], outputs=[valid, fake_A])
            self.combined_text.compile(loss=['mse', 'mae'],
                                loss_weights=[1, 100],
                                optimizer=optimizer)

    def build_generator(self, text_gan):
        """U-Net Generator"""

        def conv2d(layer_input, filters, f_size=4, bn=True):
            """Layers used during downsampling"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        def deconv2d(layer_input, skip_input, filters, f_size=4, dropout_rate=0):
            """Layers used during upsampling"""
            u = UpSampling2D(size=2)(layer_input)
            u = Conv2D(filters, kernel_size=f_size, strides=1, padding='same', activation='relu')(u)
            if dropout_rate:
                u = Dropout(dropout_rate)(u)
            u = BatchNormalization(momentum=0.8)(u)
            u = Concatenate()([u, skip_input])
            return u

        # Image input
        sketch_image = Input(shape=self.img_shape)
        input_text = Input(shape=self.text_shape)

        # Downsampling
        d1 = conv2d(sketch_image, self.gf, bn=False)
        d2 = conv2d(d1, self.gf*2)
        d3 = conv2d(d2, self.gf*4)
        d4 = conv2d(d3, self.gf*8)
        d5 = conv2d(d4, self.gf*8)
        d6 = conv2d(d5, self.gf*8)
        d7 = conv2d(d6, self.gf*8)
        d8 = conv2d(d7, self.gf*8)
        if text_gan:
            d8 = Concatenate(axis=-1)([d8, input_text])

        # Upsampling
        u1 = deconv2d(d8, d7, self.gf*8)
        u2 = deconv2d(u1, d6, self.gf*8)
        u3 = deconv2d(u2, d5, self.gf*8)
        u4 = deconv2d(u3, d4, self.gf*8)
        u5 = deconv2d(u4, d3, self.gf*4)
        u6 = deconv2d(u5, d2, self.gf+2)
        u7 = deconv2d(u6, d1, self.gf)

        u7 = UpSampling2D(size=2)(u7)
        output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

        return Model([sketch_image, input_text], output_img)

    def build_discriminator(self, text_gan):

        img_A = Input(shape=self.img_shape) 
        img_B = Input(shape=self.img_shape)
        input_text = Input(shape=(self.max_features,))
        
        def d_layer(layer_input, filters, f_size=4, bn=True, con=False):
            """Discriminator layer"""
            d = Conv2D(filters, kernel_size=f_size, strides=2, padding='same')(layer_input)
            d = LeakyReLU(alpha=0.2)(d)
            if bn:
                d = BatchNormalization(momentum=0.8)(d)
            return d

        # Concatenate image and conditioning image by channels to produce input
        combined_imgs = Concatenate(axis=-1)([img_A, img_B])

        d1 = d_layer(combined_imgs, self.df, bn=False)
        d2 = d_layer(d1, self.df*2)
        d3 = d_layer(d2, self.df*4)
        d4 = d_layer(d3, self.df*8, con=True)

        d5 = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)
        d6 = Flatten()(d5)
        if text_gan:
            d6 = Concatenate(axis=-1)([d6, input_text])
        d7 = Dense(units=1500, activation='relu')(d6)
        d7 = Dense(units=1000, activation='relu')(d7)
        validity = Dense(1, activation='sigmoid')(d7)

        return Model([img_A, img_B, input_text], validity)

    def train(self, epochs, batch_size=1, sample_interval=50):

        start_time = datetime.datetime.now()

        # Adversarial loss ground truths
        valid = np.ones((batch_size,))
        fake = np.zeros((batch_size,))

        for epoch in range(epochs):
            
            if self.simple_gan:
                self.generator_simple.save('models/generator_simple.h5')
                self.discriminator_simple.save('models/discriminator_simple.h5')
                self.combined_simple.save('models/combined_simple.h5')
            if self.text_gan:
                self.generator_text.save('models/generator_text.h5')
                self.discriminator_text.save('models/discriminator_text.h5')
                self.combined_text.save('models/combined_text.h5')

            for batch_i, (imgs_A, imgs_B, text_desc) in enumerate(self.data_loader.load_batch(batch_size)):

                # ---------------------
                #  Train Discriminator
                # ---------------------

                # Condition on B and generate a translated version
                if self.simple_gan:
                    text_desc_d = np.reshape(text_desc, newshape=(batch_size, self.max_features))
                    fake_A = self.generator_simple.predict([imgs_B, text_desc])

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real_s = self.discriminator_simple.train_on_batch([imgs_A, imgs_B, text_desc_d], valid)
                    d_loss_fake_s = self.discriminator_simple.train_on_batch([fake_A, imgs_B, text_desc_d], fake)
                    d_loss_s = 0.5 * np.add(d_loss_real_s, d_loss_fake_s)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    g_loss_s = self.combined_simple.train_on_batch([imgs_A, imgs_B, text_desc], [valid, imgs_A])

                    elapsed_time = datetime.datetime.now() - start_time
                    # Plot the progress
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss_s[0], 100*d_loss_s[1],
                                                                            g_loss_s[0],
                                                                            elapsed_time))

                if self.text_gan:
                    text_desc_d = np.reshape(text_desc, newshape=(batch_size, self.max_features))
                    fake_A = self.generator_text.predict([imgs_B, text_desc])

                    # Train the discriminators (original images = real / generated = Fake)
                    d_loss_real_t = self.discriminator_text.train_on_batch([imgs_A, imgs_B, text_desc_d], valid)
                    d_loss_fake_t = self.discriminator_text.train_on_batch([fake_A, imgs_B, text_desc_d], fake)
                    d_loss_t = 0.5 * np.add(d_loss_real_t, d_loss_fake_t)

                    # -----------------
                    #  Train Generator
                    # -----------------

                    # Train the generators
                    g_loss_t = self.combined_text.train_on_batch([imgs_A, imgs_B, text_desc], [valid, imgs_A])

                    elapsed_time = datetime.datetime.now() - start_time
                    # Plot the progress
                    print ("[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s" % (epoch, epochs,
                                                                            batch_i, self.data_loader.n_batches,
                                                                            d_loss_t[0], 100*d_loss_t[1],
                                                                            g_loss_t[0],
                                                                            elapsed_time))

                # If at save interval => save generated image samples
                if batch_i % sample_interval == 0:
                    self.sample_images(epoch, batch_i)

    def sample_images(self, epoch, batch_i):
        os.makedirs('images/%s' % self.dataset_name, exist_ok=True)
        r, c = 3, 3

        imgs_A, imgs_B, text_desc = self.data_loader.load_data(batch_size=3, is_testing=True)
        if self.simple_gan:
            fake_A_s = self.generator_simple.predict([imgs_B, text_desc])
            gen_imgs_s = np.concatenate([imgs_B, fake_A_s, imgs_A])

            # Rescale images 0 - 1
            gen_imgs_s = 0.5 * gen_imgs_s + 0.5

            # titles = ['Condition', 'Generated', 'Original']
            # fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    # axs[i,j].imshow(gen_imgs_s[cnt])
                    # axs[i, j].set_title(titles[i])
                    # axs[i,j].axis('off')
                    plt.imshow(gen_imgs_s[cnt])
                    cnt += 1
                    plt.axis('off')
                    plt.savefig("images/%s/%d_%d_%s_%d.png" % ('compare', epoch, batch_i, 'simple', cnt))
                    plt.close()

        if self.text_gan:
            fake_A_t = self.generator_text.predict([imgs_B, text_desc])

            gen_imgs_t = np.concatenate([imgs_B, fake_A_t, imgs_A])

            # Rescale images 0 - 1
            gen_imgs_t = 0.5 * gen_imgs_t + 0.5

            # titles = ['Condition', 'Generated', 'Original']
            # fig, axs = plt.subplots(r, c)
            cnt = 0
            for i in range(r):
                for j in range(c):
                    # axs[i,j].imshow(gen_imgs_t[cnt])
                    #axs[i, j].set_title(titles[i])
                    #axs[i,j].axis('off')
                    plt.imshow(gen_imgs_t[cnt])
                    cnt += 1
                    plt.axis('off')
                    plt.savefig("images/%s/%d_%d_%s_%d.png" % ('text', epoch, batch_i, 'text', cnt))
                    plt.close()


if __name__ == '__main__':
    gan = Pix2Pix(simple_gan=False, text_gan=True)
    gan.train(epochs=10, batch_size=10, sample_interval=250)