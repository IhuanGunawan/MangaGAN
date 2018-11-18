import numpy as np
from keras.layers import Input, Dense, Flatten, Conv2D, \
    LeakyReLU, Dropout, Activation, BatchNormalization, UpSampling2D, Conv2DTranspose, Reshape
from keras.models import Sequential, Model
from keras.optimizers import Adam, RMSprop
from matplotlib import pyplot as plt

HEIGHT = 256
WIDTH = 256
CHANNELS = 1
IMG_SHAPE = (WIDTH, HEIGHT, CHANNELS)
class GAN:
    def __init__(self):
        optimizer = Adam(0.0002, 0.5)
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=optimizer,
                                   metrics=['accuracy'])

        self.generator = self.build_generator()
        self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.adversarial_network = self.build_adversarial_network()
        self.adversarial_network.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])



    def build_discriminator(self):
        img_shape = (WIDTH, HEIGHT, CHANNELS)

        model = Sequential()

        model.add(Flatten(input_shape=img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        validity = model(img)

        return Model(img, validity)


    def build_generator(self):
        noise_shape = (100,)

        model = Sequential()

        model.add(Dense(256, input_shape=noise_shape))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(IMG_SHAPE), activation='tanh'))
        model.add(Reshape(IMG_SHAPE))

        model.summary()

        noise = Input(shape=noise_shape)
        img = model(noise)

        return Model(noise, img)


    def build_adversarial_network(self):
        an = Sequential()
        an.add(self.generator)
        self.discriminator.trainable = False
        an.add(self.discriminator)
        an.summary()

        return an

    def fit(self, X, epochs=2000, batch_size=256):
        half_batch = int(batch_size/2)
        for epoch in range(epochs):

            # choose a batch of rand train imgs, this is crap, do it better
            images_train = X[np.random.randint(0, len(X), half_batch)]

            # size of noise?
            noise = np.random.normal(0, 1, (half_batch, 100))
            images_gen = self.generator.predict(noise)

            # label examples
            # i should shuffle
            # train the discriminator
            x = np.concatenate((images_train, images_gen))
            y = np.ones([2*half_batch, 1])
            y[half_batch:, :] = 0
            d_loss = self.discriminator.train_on_batch(x, y)

            # train the generator
            noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
            y = np.ones([batch_size, 1])
            a_loss = self.adversarial_network.train_on_batch(noise, y)

            log_mesg = "%d: [D loss: %f, acc: %f]" % (epoch, d_loss[0], d_loss[1])
            log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])

            print(log_mesg)
            # you can save some imgs here
            if epoch % 200 == 0:
                self.save_imgs(epoch)

    def save_imgs(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].axis('off')
                cnt += 1
        # fig.savefig("gan/images/mnist_%d.png" % epoch)
        plt.show()