import os
import numpy as np
import pandas as pd
import umap.umap_ as umap

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Directory and features files
FEATURES_KAGGLE = os.path.abspath("../data/features/Kaggle")
features_CannyEdgeSpace = np.load(os.path.join(FEATURES_KAGGLE, 'features_CannyEdgeSpace_Kaggle.npy'))
features_MeanColorSpace = np.load(os.path.join(FEATURES_KAGGLE, 'features_MeanColorSpace_Kaggle.npy'))

# Compute and save UMAP embeddings
umap_CannyEdgeSpace = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_CannyEdgeSpace)
np.save('umap_CannyEdgeSpace.npy', umap_CannyEdgeSpace)
print('Kaggle/umap_CannyEdgeSpace.npy completed')

umap_MeanColorSpace = umap.UMAP(
    n_neighbors=100,
    min_dist=0.0,
    n_components=2,
    random_state=42,
).fit_transform(features_MeanColorSpace)
np.save('umap_MeanColorSpace.npy', umap_MeanColorSpace)
print('Kaggle/umap_MeanColorSpace.npy completed')

