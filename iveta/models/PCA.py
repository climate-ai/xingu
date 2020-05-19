import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Directory and features files
FEATURES_KAGGLE = os.path.abspath("../data/features/Kaggle")
features_bigearthnet = np.load(os.path.join(FEATURES_KAGGLE, 'features_bigearthnet_Kaggle.npy'))
features_EuroSAT = np.load(os.path.join(FEATURES_KAGGLE, 'features_EuroSAT_Kaggle.npy'))
features_ResNet50 = np.load(os.path.join(FEATURES_KAGGLE, 'features_ImageNet-ResNet50_Kaggle.npy'))
features_Resic45 = np.load(os.path.join(FEATURES_KAGGLE, 'features_Resic-45_Kaggle.npy'))
features_So2Sat = np.load(os.path.join(FEATURES_KAGGLE, 'features_So2Sat_Kaggle.npy'))
features_Tile2Vec = np.load(os.path.join(FEATURES_KAGGLE, 'features_Tile2Vec_Kaggle.npy'))
features_UCMerced = np.load(os.path.join(FEATURES_KAGGLE, 'features_UC_Merced_Kaggle.npy'))
features_InceptionV3 = np.load(os.path.join(FEATURES_KAGGLE, 'features_InceptionV3_pool_Kaggle.npy'))

# Compute and save PCA embeddings
pca_bigearthnet = PCA(
    n_components=17
).fit_transform(features_bigearthnet)
np.save('pca_bigearthnet.npy', pca_bigearthnet)
print('Kaggle/pca_bigearthnet.npy completed')

pca_EuroSAT = PCA(
    n_components=17
).fit_transform(features_EuroSAT)
np.save('pca_EuroSAT.npy', pca_EuroSAT)
print('Kaggle/pca_EuroSAT.npy completed')

pca_ResNet50 = PCA(
    n_components=17
).fit_transform(features_ResNet50)
np.save('pca_ResNet50.npy', pca_ResNet50)
print('Kaggle/pca_ResNet50.npy completed')

pca_Resic45 = PCA(
    n_components=17
).fit_transform(features_Resic45)
np.save('pca_Resic45.npy', pca_Resic45)
print('Kaggle/pca_Resic45.npy completed')


pca_So2Sat = PCA(
    n_components=17
).fit_transform(features_So2Sat)
np.save('pca_So2Sat.npy', pca_So2Sat)
print('Kaggle/pca_So2Sat.npy completed')

pca_Tile2Vec = PCA(
    n_components=17
).fit_transform(features_Tile2Vec)
np.save('pca_Tile2Vec.npy', pca_Tile2Vec)
print('Kaggle/pca_Tile2Vec.npy completed')

pca_UCMerced = PCA(
    n_components=17
).fit_transform(features_UCMerced)
np.save('pca_UCMerced.npy', pca_UCMerced)
print('Kaggle/pca_UCMerced.npy completed')

pca_InceptionV3 = PCA(
    n_components=17
).fit_transform(features_InceptionV3)
np.save('pca_InceptionV3.npy', pca_InceptionV3)
print('Kaggle/pca_InceptionV3.npy completed')


