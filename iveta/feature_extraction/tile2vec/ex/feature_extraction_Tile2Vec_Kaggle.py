import os
import numpy as np
import cv2
import torch
from torch.autograd import Variable

import sys
sys.path.append('../')
from src.tilenet import make_tilenet
from src.resnet import ResNet18

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used


# Import model and extract features
# Setting up model
in_channels = 4
z_dim = 512
cuda = torch.cuda.is_available()
tilenet = ResNet18()
if cuda: tilenet.cuda()

# Load parameters
model_fn = '../models/naip_trained.ckpt'
#checkpoint = torch.load(model_fn, map_location='cpu')
checkpoint = torch.load(model_fn)
tilenet.load_state_dict(checkpoint)
tilenet.eval()

# Get data
tile_dir = '../../../data/train-tif-v2'
n_tiles = 40479

# Embed tiles
X = np.zeros((n_tiles, z_dim))
for idx in range(n_tiles):
    tile_raw = cv2.imread(os.path.join(tile_dir, 'train_{}.tif'.format(idx)), cv2.IMREAD_UNCHANGED)
    tile_raw = cv2.resize(tile_raw, (100, 100))
    # Get 4 channels (BGR & NIR)
    tile_raw = tile_raw[:,:,:4]
    # Switch channels to RGB &  NIR order
    tile = np.zeros(tile_raw.shape)
    tile[:,:,0] = tile_raw[:,:,2]
    tile[:,:,1] = tile_raw[:,:,1]
    tile[:,:,2] = tile_raw[:,:,0]
    tile[:,:,3] = tile_raw[:,:,3]
    # Rearrange to PyTorch order
    tile = np.moveaxis(tile, -1, 0)
    tile = np.expand_dims(tile, axis=0)
    # Scale to [0, 1]
    tile = tile / 39786
    # Embed tile
    tile = torch.from_numpy(tile).float()
    tile = Variable(tile)
    if cuda: tile = tile.cuda()
    z = tilenet.encode(tile)
    if cuda: z = z.cpu()
    z = z.data.numpy()
    X[idx,:] = z
    
print('Embedded {} tiles.'.format(n_tiles))

# Export features as numpy array
np.save('../../data/features/Kaggle/features_Tile2Vec_Kaggle.npy', X)

# Export featuers in csv file
#np.savetxt('features' + model_name + '_Kaggle.csv', features, delimiter=',')




