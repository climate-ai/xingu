import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

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

# Compute and save TSNE embeddings
tsne_bigearthnet = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_bigearthnet)
np.save('tsne_bigearthnet.npy', tsne_bigearthnet)
print('Kaggle/tsne_bigearthnet.npy completed')

tsne_EuroSAT = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_EuroSAT)
np.save('tsne_EuroSAT.npy', tsne_EuroSAT)
print('Kaggle/tsne_EuroSAT.npy completed')

tsne_ResNet50 = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_ResNet50)
np.save('tsne_ResNet50.npy', tsne_ResNet50)
print('Kaggle/tsne_ResNet50.npy completed')

tsne_Resic45 = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_Resic45)
np.save('tsne_Resic45.npy', tsne_Resic45)
print('Kaggle/tsne_Resic45.npy completed')


tsne_So2Sat = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_So2Sat)
np.save('tsne_So2Sat.npy', tsne_So2Sat)
print('Kaggle/tsne_So2Sat.npy completed')

tsne_Tile2Vec = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_Tile2Vec)
np.save('tsne_Tile2Vec.npy', tsne_Tile2Vec)
print('Kaggle/tsne_Tile2Vec.npy completed')

tsne_UCMerced = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_UCMerced)
np.save('tsne_UCMerced.npy', tsne_UCMerced)
print('Kaggle/tsne_UCMerced.npy completed')

tsne_InceptionV3 = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_InceptionV3)
np.save('tsne_InceptionV3.npy', tsne_InceptionV3)
print('Kaggle/tsne_InceptionV3.npy completed')


