import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense
from tqdm import tqdm
from sklearn.metrics import fbeta_score

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Directories
PLANET_KAGGLE_ROOT = os.path.abspath("../data/")
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

# Prepare array of images of labels for higher resolution
x_train = []
y_train = []

for f, tags in tqdm(labels_df.values, miniters=1000):
    img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1
    x_train.append(cv2.resize(img, (224, 224)))
    y_train.append(targets)

# Normalize images into the interval [0,1]
y_train = np.array(y_train, np.uint8)
x_train = np.array(x_train, np.float16) / 255.

# Split between training and validation sets
# Total images in dataset: 40 478 images
SPLIT_TRAIN = 10000
SPLIT_VALIDATE = 12000
x_train, x_valid, x_test, y_train, y_valid = ( \
        x_train[:SPLIT_TRAIN], x_train[SPLIT_TRAIN:SPLIT_VALIDATE], x_train[SPLIT_VALIDATE:], \
        y_train[:SPLIT_TRAIN], y_train[SPLIT_TRAIN:SPLIT_VALIDATE])

# Import remote sensing module and extract features

module = hub.KerasLayer("https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1")

model = tf.keras.Sequential([
                             module,
                             tf.keras.layers.Dense(17, activation="sigmoid"),
                             ])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=10,
                    verbose=1,
                    validation_data=(x_valid, y_valid))

# Save model
model.save('bigearthnet-resnet_model_v05.h5')
model.save_weights('bigearthnet-resnet_model_v05_weights.h5')
#tf.keras.experimental.export_saved_model(model, 'bigearthnet-resnet_model_v04_exp.h5')

# Calculate and print validation set F-Beta score
p_valid = model.predict(x_valid, batch_size=128)
np.savetxt('features_bigearth_v05_p_valid.csv', p_valid, delimiter=',')
print(fbeta_score(y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))

# Predict labels for test data
p_test = model.predict(x_test, batch_size=128)
np.savetxt('features_bigearth_v05_p_test.csv', p_test, delimiter=',')
