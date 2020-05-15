import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import tensorflow_hub as hub
from keras.models import Sequential
from tqdm import tqdm

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Load Kaggle images
try:
    images = np.load('train_images_Kaggle.npy')
except:
    # Directory
    PLANET_KAGGLE_ROOT = os.path.abspath("../data/")
    PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
    PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
    assert os.path.exists(PLANET_KAGGLE_ROOT)
    assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
    assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)

    # Read in image ids and tags
    # Total images in dataset: 40 479 images
    labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)

    # Preprocess images into 224x224 resolution higher resolution
    images = []
    for f, tags in tqdm(labels_df.values, miniters=1000):
        img = cv2.imread('../data/train-jpg/{}.jpg'.format(f))
        images.append(cv2.resize(img, (224, 224)))

    # Normalize images into the interval [0,1]
    images = np.array(images, np.float16) / 255.
    
    # Save images as numpy array
    np.save('train_images_Kaggle.npy', images)


def export_features(features, model_name):
    # Export features as numpy array
    np.save('features_'+ model_name +'_Kaggle.npy', features)
    # Export featuers in csv file
    #np.savetxt('features' + model_name + '_Kaggle.csv', features, delimiter=',')


def extract_and_export_features(images, model_name, path, batch_size=128):
    # Import remote sensing module and extract features
    module = hub.KerasLayer(path)
    model = tf.keras.Sequential([module])
    features = model.predict(images, batch_size=128)
    export_features(features, model_name)


#Export features for various remote sensing models
extract_and_export_features(images, 'BigEarthNet',\
    "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1")

extract_and_export_features(images, 'EuroSAT',\
    "https://tfhub.dev/google/remote_sensing/eurosat-resnet50/1")

extract_and_export_features(images, 'Resic-45',\
    "https://tfhub.dev/google/remote_sensing/resisc45-resnet50/1")
    
extract_and_export_features(images, 'So2Sat',\
    "https://tfhub.dev/google/remote_sensing/so2sat-resnet50/1")
    
extract_and_export_features(images, 'UC_Merced',\
    "https://tfhub.dev/google/remote_sensing/uc_merced-resnet50/1")

