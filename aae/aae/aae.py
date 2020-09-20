from __future__ import print_function, division
import tensorflow.keras as keras
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from tensorflow.keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import tensorflow.keras.backend as K
import numpy as np
class AdversarialAutoencoder():
    def __init__(self,Model_mode=1):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 10

        optimizer = Adam(0.001, 0.5)
        self.Model_mode=Model_mode
        

        # Build the encoder / decoder
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

        img = Input(shape=self.img_shape)
        # The generator takes the image, encodes it and reconstructs it
        # from the encoding
        encoded_repr = self.encoder(img)
        reconstructed_img = self.decoder(encoded_repr)
        # For the adversarial_autoencoder model we will only train the generator  
        
        if(self.Model_mode==1):                 #!使用潜在编码对抗
            self.discriminator = self.build_discriminator(input_shape=self.latent_dim)
            self.discriminator.compile(loss='binary_crossentropy',
                                        # Build and compile the discriminator
                                        optimizer=optimizer,
                                        metrics=['accuracy'])
            self.discriminator.trainable = False
        # The discriminator determines validity of the encoding
            validity = self.discriminator(encoded_repr)
        else:                                   #!使用生成图片对抗
            self.discriminator = self.build_discriminator(input_shape=np.prod(self.img_shape))
            self.discriminator.compile(loss='binary_crossentropy',
                                        # Build and compile the discriminator
                                        optimizer=optimizer,
                                        metrics=['accuracy'])
            self.discriminator.trainable = False
            reconstructed_img= Flatten()(reconstructed_img)
            validity=self.discriminator(reconstructed_img)
        # The adversarial_autoencoder model  (stacked generator and discriminator)
        self.adversarial_autoencoder = Model(img, [reconstructed_img, validity])
        self.adversarial_autoencoder.compile(loss=['mse', 'binary_crossentropy'],
            loss_weights=[0.999, 0.001],
            optimizer=optimizer)


    def build_encoder(self):
        # Encoder

        img = Input(shape=self.img_shape)

        h = Flatten()(img)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        h = Dense(512)(h)
        h = LeakyReLU(alpha=0.2)(h)
        mu = Dense(self.latent_dim)(h)
        #log_var = Dense(self.latent_dim)(h)
        #latent_repr = merge([mu, log_var],
            #    mode=lambda p: p[0] + K.random_normal(K.shape(p[0])) * K.exp(p[1] / 2),
            #    output_shape=lambda p: p[0])

        return Model(img, mu)

    def build_decoder(self):

        model = Sequential()

        model.add(Dense(512, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        model.summary()

        z = Input(shape=(self.latent_dim,))
        img = model(z)

        return Model(z, img)

    def build_discriminator(self,input_shape):

        model = Sequential()

        model.add(Dense(512, input_dim=input_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation="sigmoid"))
        model.summary()
        encoded_repr = Input(shape=(input_shape, ))
        validity = model(encoded_repr)

        return Model(encoded_repr, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # Load the dataset
        (X_train, _), (_, _) = mnist.load_data()

        # Rescale -1 to 1
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            if(self.Model_mode==1):
                img_fake = self.encoder.predict(imgs)
                img_real = np.random.normal(size=(batch_size, self.latent_dim))
            else:
                letent=self.encoder.predict(imgs)
                img_fake=self.decoder.predict(letent)
                img_real=Flatten()(imgs)
                
                img_fake=Flatten()(imgs)

            # Train the discriminator
            d_loss_real = self.discriminator.train_on_batch(img_real, valid)
            d_loss_fake = self.discriminator.train_on_batch(img_fake, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            # ---------------------
            #  Train Generator
            # ---------------------
            # Train the generator
            g_loss = self.adversarial_autoencoder.train_on_batch(imgs, [img_fake, valid])

            # Plot the progress
            print ("%d [D loss: %f, acc: %.2f%%] [G loss: %f, mse: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss[0], g_loss[1]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5

        z = np.random.normal(size=(r*c, self.latent_dim))
        gen_imgs = self.decoder.predict(z)

        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("img_mnist_%d.png" % epoch)
        plt.close()

    def save_model(self):

        def save(model, model_name):
            model_path = "saved_model/%s.json" % model_name
            weights_path = "saved_model/%s_weights.hdf5" % model_name
            options = {"file_arch": model_path,
                        "file_weight": weights_path}
            json_string = model.to_json()
            open(options['file_arch'], 'w').write(json_string)
            model.save_weights(options['file_weight'])

        save(self.generator, "aae_generator")
        save(self.discriminator, "aae_discriminator")
if __name__ == '__main__':
    aae = AdversarialAutoencoder(Model_mode=0)
    with tf.device("/gpu:0"):
        aae.train(epochs=2000, batch_size=32, sample_interval=99)
class AdversarialAutoencoder_END2END():
    def __init__(self):
        super().__init__()
        self.latent_dim=10
        optimizer=keras.optimizers.Adam(0.0002,0.5)
        self.img_shape=[28,28,3]

    def img_encoder(self,img_shape,latent_dim):
        img=keras.layers.Input(shape=img_shape)
        h=keras.layers.Flatten()
        h=keras.layers.Dense(units=512)(h)
        h=keras.layers.LeakyReLU(alpha=0.2)(h)
        h=keras.layers.Dense(units=512)(h)
        h=keras.layers.LeakyReLU(alpha=0.2)(h)
        mu=keras.layers.Dense(latent_dim)(h)
        log_var=keras.layers.merge([mu,log_var],
                                mode=lambda p:p[0]+K.random_normal(K.shape(p[0]))*K.exp(p[1]/2),
                                output_shape=lambda p:p[0])
        return keras.Model(img,latent_dim)
        
    def img_decoder(self,img_shape,latent_dim):
        z=keras.layers.Input(shape=(latent_dim,))
        h=keras.layers.Dense(512)(z)
        h=keras.layers.LeakyReLU(alpha=0.2)(h)
        h=keras.layers.Dense(512)(h)
        h=keras.layers.LeakyReLU(alpha=0.2)(h)
        h=keras.layers.Dense(np.prod(img_shape),activation='tanh')(h)
        img=keras.layers.Reshape(img_shape)(h)

        return keras.Model(z,img)
    
    #def discriminator(self):
        

