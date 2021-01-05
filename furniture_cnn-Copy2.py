#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# MODIFY THE DATASET BY RUNNING NURAN'S CODE


# In[ ]:


# SPLITTING THE DATASET INTO TRAIN, TEST AND VALIDATION SET
import os
import splitfolders
inp_folder = "/media/arpan/282c8097-5a01-4850-9251-18491bc989db/arpan/Downloads/cumulative_furniture_types"
os.mkdir("/media/arpan/282c8097-5a01-4850-9251-18491bc989db/arpan/Downloads/seperated_cumulative_furniture_types")
out_folder = "/media/arpan/282c8097-5a01-4850-9251-18491bc989db/arpan/Downloads/seperated_cumulative_furniture_types" 
splitfolders.ratio(inp_folder, out_folder, seed = 42, ratio = (.6, .2, .2))


# In[ ]:


# import the required libraries

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np 
import sklearn
from sklearn.metrics import confusion_matrix
import itertools

# path for the test, train and validation data

train_dir = "/media/arpan/282c8097-5a01-4850-9251-18491bc989db/arpan/Downloads/seperated_cumulative_furniture_types/train"
test_dir = "/media/arpan/282c8097-5a01-4850-9251-18491bc989db/arpan/Downloads/seperated_cumulative_furniture_types/test"
val_dir = "/media/arpan/282c8097-5a01-4850-9251-18491bc989db/arpan/Downloads/seperated_cumulative_furniture_types/val"


# using transfer learning with the pretrained inception v3 model with imagenet weights.
# fixing the weight of the lower layers and we dont want to change this

pre_trained_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')
for layer in pre_trained_model.layers:
    layer.trainable = False
    
# setting parameters 
            
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta=0.001)

checkpoint = ModelCheckpoint('furniture_detection.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)

earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=10, verbose=1, restore_best_weights=True)

callbacks = [reduce_lr, checkpoint]

# modifying the top layers for six output classes i.e. bed, chair, table, lamp, sofa and dresser
            
x = layers.Flatten()(pre_trained_model.output)
x = layers.Dense(1024, activation = 'relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(6, activation='sigmoid')(x)

# compiling the model, while we could also select other optimizers and choose the best suited, while loss is categorical_crossentropy, as we are dealing with multi class data

model = Model(pre_trained_model.input, x)
model.compile(optimizer = RMSprop(lr=0.0001), loss = 'categorical_crossentropy', metrics = ['acc'])

# augumenting the image data for better learning

train_datagen = ImageDataGenerator(rescale = 1/255, rotation_range = 40, width_shift_range = 0.2, height_shift_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1/255)
datagen_validation = ImageDataGenerator(rescale = 1/255)
train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 100, class_mode = 'categorical', target_size = (150, 150))
test_generator = test_datagen.flow_from_directory(test_dir, batch_size = 100, class_mode = 'categorical', target_size = (150, 150))
val_generator = datagen_validation.flow_from_directory(val_dir, target_size=(150, 150), class_mode='categorical')
history = model.fit_generator(train_generator, validation_data = test_generator, steps_per_epoch = 146, epochs = 500, validation_steps = 49, verbose = 1, callbacks=[callbacks])


# In[ ]:


plt.figure(figsize=(30,10))
plt.subplot(1, 2, 1)
plt.ylabel('Loss', fontsize=16)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')

plt.subplot(1, 2, 2)
plt.ylabel('Accuracy', fontsize=16)
plt.plot(history.history['acc'], label='Training Accuracy')
plt.plot(history.history['val_acc'], label='Validation Accuracy')
plt.show()


# In[ ]:


predictions = model.predict_generator(generator=val_generator)
y_pred = [np.argmax(pb) for pb in predictions]
y_test = val_generator.classes
class_names = val_generator.class_indices.keys()
print(class_names)

def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    plt.figure(figsize=(10,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
# compute confusion matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)

# plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, title='Normalized confusion matrix')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




