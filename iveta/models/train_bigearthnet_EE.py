import os
import numpy as np
import pandas as pd
import cv2
#import seaborn as sns
import tensorflow as tf
import tensorflow_hub as hub
from keras.utils.np_utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # select ID of GPU that shall be used

# Directories
EE_DATA_ROOT = os.path.abspath("../data/EE_data/")
assert os.path.exists(EE_DATA_ROOT)

# Path to original remote sensing module
path_to_module = "https://tfhub.dev/google/remote_sensing/bigearthnet-resnet50/1"

# Load numpy Sentinel data with labels
x_train = np.load(os.path.join(EE_DATA_ROOT, 'x_train.npy'))
y_train = np.load(os.path.join(EE_DATA_ROOT, 'y_train.npy'))

x_validate = np.load(os.path.join(EE_DATA_ROOT, 'x_validate.npy'))
y_validate = np.load(os.path.join(EE_DATA_ROOT, 'y_validate.npy'))

# Normalize x data to values [0,1] based on original Google EE export
x_train[x_train > 0.3] = 0.3
x_train = x_train / 0.3

x_validate[x_validate > 0.3] = 0.3
x_validate = x_validate / 0.3

# Reshape x data to square format, use only rgb channels
img_res = np.array([7,7])
x_train = x_train[:,:49*3].reshape(x_train.shape[0], 7, 7, 3)
x_validate = x_validate[:,:49*3].reshape(x_validate.shape[0], 7, 7, 3)

# Switch channel order from BGR to RGB
temp = np.copy(x_train[:,:,:,0])
x_train[:,:,:,0] = x_train[:,:,:,2]
x_train[:,:,:,2] = temp

temp = np.copy(x_validate[:,:,:,0])
x_validate[:,:,:,0] = x_validate[:,:,:,2]
x_validate[:,:,:,2] = temp

# Resize image patches to original module resolution
height = 224
width = 224

x_train_resized = np.zeros((x_train.shape[0], height, width, 3))
for i in range(x_train.shape[0]):
    x_train_resized[i, ...]= cv2.resize(x_train[i, ...], (height, width))

x_validate_resized = np.zeros((x_validate.shape[0], height, width, 3))
for i in range(x_validate.shape[0]):
    x_validate_resized[i, ...]= cv2.resize(x_validate[i, ...], (height, width))
    
x_train = x_train_resized
x_validate = x_validate_resized

# Convert y data to one-hot vector
y_train = to_categorical(y_train[:,0], num_classes=10) #use only tree cover label
y_validate = to_categorical(y_validate[:,0], num_classes=10) #use only tree cover label

# Import remote sensing module and extract features
module = hub.KerasLayer(path_to_module)

model = tf.keras.Sequential([
                             module,
                             tf.keras.layers.Dense(10, activation="softmax"),
                             ])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model
filepath="trained/bigearthnet-resnet_weights_EE_v01.hdf5"
checkpointer = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = model.fit(x_train, y_train, #train for tree cover classification only
                    batch_size=128,
                    epochs=10,
                    callbacks=[checkpointer],
                    verbose=1,
                    validation_data=(x_validate, y_validate))

# Save model
model.save('trained/bigearthnet-resnet_EE_model_v01.hdf5')

# Print accuracy and validation accuracy
df = pd.DataFrame({'epochs':history.epoch, 'accuracy': history.history['acc'], 'validation_accuracy': history.history['val_acc']})
print(df.head(10))
#g = sns.pointplot(x="epochs", y="accuracy", data=df, fit_reg=False)
#g = sns.pointplot(x="epochs", y="validation_accuracy", data=df, fit_reg=False, color='green')

# Calculate and print validation set accuracy score
p_validate = model.predict(x_validate, batch_size=128)
np.savetxt('../predictions/bigearthnet-resnet_p_validate_v01.csv', p_validate)
print(accuracy_score(y_validate.argmax(axis=1), p_validate.argmax(axis=1)))

# Print confusion matrix
print(confusion_matrix(y_validate.argmax(axis=1), p_validate.argmax(axis=1)))
