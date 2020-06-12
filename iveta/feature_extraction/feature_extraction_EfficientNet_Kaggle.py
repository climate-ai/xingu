import os
import numpy as np
from PIL import Image
from efficientnet_pytorch import EfficientNet
import torch
from torchvision import transforms

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"  # select ID of GPU that shall be used

# Setting up model
model = EfficientNet.from_pretrained('efficientnet-b0')
model.eval()

# Get data
img_dir = '../data/train-jpg'
n_img = 40479
z_dim = 7*7*1280

# Embed tiles
X = np.zeros((n_img, z_dim))
for idx in range(n_img):
    # Read in and preprocess RGB image
    path = os.path.join(img_dir, 'train_{}.jpg'.format(idx))
    tfms = transforms.Compose([transforms.Resize(224), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),])
    img = tfms(Image.open(path).convert('RGB')).unsqueeze(0)
    #if idx==0:
    #    print(img.shape) # torch.Size([1, 3, 224, 224])
    # Embed img
    z = model.extract_features(img)
    z = z.detach().numpy()
    z = z.flatten()
    X[idx,:] = z
    if (idx % 1000) == 0:
        print(idx)
    
print('Embedded {} images.'.format(n_img))

# Export features as numpy array
np.save('../data/features/Kaggle/features_EfficientNet_Kaggle.npy', X)

# Export featuers in csv file
#np.savetxt('features' + model_name + '_Kaggle.csv', features, delimiter=',')




