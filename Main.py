from GAN import GAN
from os import listdir
import cv2
import numpy as np
#this should be using the constants
data = np.array([cv2.imread('./input/' + f, 0) for f in listdir('./input')]).reshape((-1, 256,256,1))
# resize the data
# clean up
gan = GAN()
gan.fit(data)