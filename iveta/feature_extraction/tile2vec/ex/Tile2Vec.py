import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_hub as hub
import torch
from torch.autograd import Variable
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from tqdm import tqdm
from sklearn.metrics import fbeta_score

import sys
sys.path.append('../')
from src.tilenet import make_tilenet
from src.resnet import ResNet18

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Directories
PLANET_KAGGLE_ROOT = os.path.abspath("../../data/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

# Read in labels
labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

# Build list with unique labels
label_list = []
for tag_str in labels_df.tags.values:
    labels = tag_str.split(' ')
    for label in labels:
        if label not in label_list:
            label_list.append(label)

# Preprocess labels
label_map = {l: i for i, l in enumerate(label_list)}
inv_label_map = {i: l for l, i in label_map.items()}

# Prepare array of images of labels for higher resolution
x_train_HR = []
y_train_HR = []

for f, tags in tqdm(labels_df.values, miniters=1000):
    img = cv2.imread('../../data/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train_HR.append(cv2.resize(img, (512, 512)))
    y_train_HR.append(targets)

# Normalize images into the interval [0,1]
y_train_HR = np.array(y_train_HR, np.uint8)
x_train_HR = np.array(x_train_HR, np.float16) / 255.

# Split between training and validation sets
split = 1000  # Total number: 40 478 images
split_2 = 40478
x_train_HR, x_valid_HR, y_train_HR, y_valid_HR = (x_train_HR[:split], x_train_HR[split:split_2], y_train_HR[:split], y_train_HR[split:split_2])

# Import remote sensing module and extract features

# Setting up model
in_channels = 4
z_dim = 512
cuda = torch.cuda.is_available()
# tilenet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# Use old model for now
tilenet = ResNet18()
if cuda: tilenet.cuda()

# Load parameters
model_fn = '../models/naip_trained.ckpt'
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)

model = tf.keras.Sequential([
                             tilenet,
                             tf.keras.layers.Dense(17, activation="sigmoid"),
                             ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(x_train_HR, y_train_HR,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_valid_HR, y_valid_HR))
# Save model
model.save('Tile2Vec-resnet_model_v01.h5')

# Use the model to predict
p_valid = model.predict(x_valid_HR, batch_size=128)
np.savetxt('features_Tile2Vec_v01_predict.csv', p_valid, delimiter=',')

# Print validation set F-Beta score
print(fbeta_score(y_valid_HR, np.array(p_valid) > 0.2, beta=2, average='samples'))

#features = module(x_valid_HR)  # Features with shape [batch_size, num_features]
#print(features)
#with tf.Session() as sess:
#    tf.global_variables_initializer().run()
#    o = sess.run(features)
#    o = np.add(o, 0)
#    print(type(o))
#    print(o)
#    np.savetxt('features_bigearth_v01.csv', o, delimiter=',')
