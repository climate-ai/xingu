import os
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Directory and features files
FEATURES_KAGGLE = os.path.abspath("../data/features/Kaggle")
features_CannyEdgeSpace = np.load(os.path.join(FEATURES_KAGGLE, 'features_CannyEdgeSpace_Kaggle.npy'))
features_MeanColorSpace = np.load(os.path.join(FEATURES_KAGGLE, 'features_MeanColorSpace_Kaggle.npy'))

# Compute and save PCA embeddings
pca_CannyEdgeSpace = PCA(
    n_components=2
).fit_transform(features_CannyEdgeSpace)
np.save('pca_CannyEdgeSpace.npy', pca_CannyEdgeSpace)
print('Kaggle/pca_CannyEdgeSpace.npy completed')

pca_MeanColorSpace = PCA(
    n_components=2
).fit_transform(features_MeanColorSpace)
np.save('pca_MeanColorSpace.npy', pca_MeanColorSpace)
print('Kaggle/pca_MeanColorSpace.npy completed')

