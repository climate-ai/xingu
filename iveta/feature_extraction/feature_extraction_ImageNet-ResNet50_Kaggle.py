import os
import numpy as np
import cv2
from keras.applications import ResNet50
from keras.applications.resnet50 import preprocess_input

#from keras.applications import ResNet50V2
#from keras.applications.resnet_v2 import preprocess_input

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Setting up model
model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
model.summary()

# Get data
img_dir = '../data/train-jpg'
n_img = 40479
z_dim = 2048
input_shape = (299, 299)

# Embed tiles
X = np.zeros((n_img, z_dim))
for idx in range(n_img):
    # Read in RGB image
    img = cv2.imread(os.path.join(img_dir, 'train_{}.jpg'.format(idx)))
    img = cv2.resize(img, input_shape)
    # Expand shape
    img = np.expand_dims(img, axis=0)
    if idx==0:
        print(img.shape)
    # Preprocess mean and stdev
    img = preprocess_input(img)
    # Embed img
    z = model.predict(img)
    if idx==0:
        print(z.shape)
    z = np.array(z).flatten()
    if idx==0:
        print(z.shape)
        print(z)
    X[idx,:] = z
    if (idx % 1000) == 0:
        print(idx)
    
print('Embedded {} images.'.format(n_img))

# Export features as numpy array
np.save('../data/features/Kaggle/features_ImageNet-ResNet50_Kaggle.npy', X)

# Export featuers in csv file
#np.savetxt('features' + model_name + '_Kaggle.csv', features, delimiter=',')




