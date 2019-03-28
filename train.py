import keras
import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from utility import *
import os
import tensorflow as tf
from keras.backend import tensorflow_backend
import time
import random
from keras.callbacks import TensorBoard

config = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
session = tf.Session(config=config)
tensorflow_backend.set_session(session)


content_dir = '/Users/user/AuDev/artistic_style_transfer/fast_style_transfer/content/'
style_dir = '/Users/user/AuDev/artistic_style_transfer/fast_style_transfer/style'
output_dir = '/Users/user/AuDev/artistic_style_transfer/fast_style_transfer/starry_night_output'

height = 256
width = 256

vgg_net = get_vgg(height,width)

model = get_full_network(height, width)
model.compile(optimizer='adam',
              loss=[TVLoss, styleLoss1, styleLoss3, styleLoss3, styleLoss4, featureLoss],
              loss_weights=[1, 1, 1, 1, 1, 1e-5])
model.summary()
style_images = []
style_names = os.listdir(style_dir)
for name in style_names:
    img = image.load_img(style_dir+'/'+name, target_size=(height,width))
    x = image.img_to_array(img)
    style_images.append(x)

style_images_array = np.stack(style_images)
sf = vgg_net.predict(style_images_array, batch_size=1)
print(sf[0].shape)
print(sf[1].shape)
print(sf[2].shape)
print(sf[3].shape)
print(len(sf))
print(sf[0][0:1,:,:,:].shape)

datagen = ImageDataGenerator()

k = 0
bsize = 1
epochs = 20000
ep = 0
for x in datagen.flow_from_directory(content_dir, batch_size=bsize, class_mode=None, target_size=(height,width), ):
    # print(x.shape)
    if ep%10 == 0:
        print(str(ep))
    # print(x.shape)
    cf = vgg_net.predict(x, batch_size=1)[2]
    # print("sf[0][k:k+bsize,:,:,:]={}, k={}".format(sf[0][k:k+bsize,:,:,:].shape, k))
    # print("sf[1][k:k+bsize,:,:,:]={}, k={}".format(sf[1][k:k + bsize, :, :, :].shape, k))
    # print("sf[2][k:k+bsize,:,:,:]={}, k={}".format(sf[2][k:k + bsize, :, :, :].shape, k))
    # print("sf[3][k:k+bsize,:,:,:]={}, k={}".format(sf[3][k:k + bsize, :, :, :].shape, k))
    print("**************************************************")
    print("ep = {}".format(ep))
    history = model.train_on_batch(x, [x, sf[0][k:k+bsize,:,:,:], sf[1][k:k+bsize,:,:,:], sf[2][k:k+bsize,:,:,:], sf[3][k:k+bsize,:,:,:], cf])
    print("model metrics :{}  value = {}".format(model.metrics_names,history))
    k+=1
    if k+bsize >= len(style_images):
        k=0
    ep+=1
    if ep>epochs:
        break

model.save("./tmp20k.h5")


"""
content_images_array = np.load(input_dir + '/images.npy')
cf = np.load(input_dir + '/cf.npy')
sf1 = np.load(input_dir + '/sf1.npy')
sf2 = np.load(input_dir + '/sf2.npy')
sf3 = np.load(input_dir + '/sf3.npy')
sf4 = np.load(input_dir + '/sf4.npy')
tb_cb = keras.callbacks.TensorBoard(log_dir="./log", histogram_freq=100)
model = getWholeNet(height, width)
model.compile(optimizer='adam', 
              loss=[TVLoss, styleLoss1, styleLoss3, styleLoss3, styleLoss4, featureLoss],
              loss_weights=[1, 1, 1, 1, 1, 1e-5])
model.fit(content_images_array, [content_images_array, sf1, sf2, sf3, sf4, cf], 
          batch_size=4, epochs=100, callbacks=[tb_cb])
model.save(output_dir + '/star1000.h5')
"""
