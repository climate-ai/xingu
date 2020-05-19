import os
import numpy as np
import pandas as pd
import umap.umap_ as umap

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

# Compute and save UMAP embeddings
umap_bigearthnet = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_bigearthnet)
np.save('umap_bigearthnet.npy', umap_bigearthnet)
print('Kaggle/umap_bigearthnet.npy completed')

umap_EuroSAT = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_EuroSAT)
np.save('umap_EuroSAT.npy', umap_EuroSAT)
print('Kaggle/umap_EuroSAT.npy completed')

umap_ResNet50 = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_ResNet50)
np.save('umap_ResNet50.npy', umap_ResNet50)
print('Kaggle/umap_ResNet50.npy completed')

umap_Resic45 = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_Resic45)
np.save('umap_Resic45.npy', umap_Resic45)
print('Kaggle/umap_Resic45.npy completed')


umap_So2Sat = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_So2Sat)
np.save('umap_So2Sat.npy', umap_So2Sat)
print('Kaggle/umap_So2Sat.npy completed')

umap_Tile2Vec = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_Tile2Vec)
np.save('umap_Tile2Vec.npy', umap_Tile2Vec)
print('Kaggle/umap_Tile2Vec.npy completed')

umap_UCMerced = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_UCMerced)
np.save('umap_UCMerced.npy', umap_UCMerced)
print('Kaggle/umap_UCMerced.npy completed')

umap_InceptionV3 = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_InceptionV3)
np.save('umap_InceptionV3.npy', umap_InceptionV3)
print('Kaggle/umap_InceptionV3.npy completed')


