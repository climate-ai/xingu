import os
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Directory and features files
FEATURES_KAGGLE = os.path.abspath("../data/features/Kaggle")
features_CannyEdgeSpace = np.load(os.path.join(FEATURES_KAGGLE, 'features_CannyEdgeSpace_Kaggle.npy'))
features_MeanColorSpace = np.load(os.path.join(FEATURES_KAGGLE, 'features_MeanColorSpace_Kaggle.npy'))

# Compute and save TSNE embeddings
tsne_CannyEdgeSpace = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_CannyEdgeSpace)
np.save('tsne_CannyEdgeSpace.npy', tsne_CannyEdgeSpace)
print('Kaggle/tsne_CannyEdgeSpace.npy completed')

tsne_MeanColorSpace = TSNE(
    n_components=2,
    init='random', # pca
    random_state=101,
    method='barnes_hut',
    n_iter=500,
    verbose=2
).fit_transform(features_MeanColorSpace)
np.save('tsne_MeanColorSpace.npy', tsne_MeanColorSpace)
print('Kaggle/tsne_MeanColorSpace.npy completed')
