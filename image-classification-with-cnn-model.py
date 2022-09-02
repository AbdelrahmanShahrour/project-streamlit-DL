#!/usr/bin/env python
# coding: utf-8

# # Description
# ### This dataset contains 6,899 images from 8 distinct classes compiled from various sources (see Acknowledgements). The classes include airplane, car, cat, dog, flower, fruit, motorbike and person.

# # import lib

# In[2]:


from IPython.display import Image, display
# preprocessing and processing
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
# ploting
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from tensorflow.keras.utils import plot_model
# split data
from sklearn.model_selection import train_test_split
# CNN
from keras import models, layers
# val
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[38]:


import os

labels = os.listdir('..\Computer Vision\imagedata\data\\natural_images')
print(labels)


# # show some images

# In[43]:


num = []
for label in labels:
    path = "..\Computer Vision\imagedata\data\\natural_images\{}\\".format(label)
    
    folder_data = os.listdir(path)
    k = 0
    print('\n',f'=====   {label.upper()}   =====')
    for image_path in folder_data:
        if k < 5:
            display(Image(path+image_path))
        k = k+1
    num.append(k)
    print(f'count : {k} images , label : {label} class')


# # vis count image for all classes

# In[44]:


fig = go.Figure(data=[go.Bar(
            x=labels, y=num,
            text=num,
            textposition='auto',
        )])
fig.update_layout(title_text='NUMBER OF IMAGES CONTAINED IN EACH CLASS')
fig.show()


# # show shaps and stadardizing input data

# In[46]:


x_data =[]
y_data = []
import cv2
for label in labels:
    path = "..\Computer Vision\imagedata\data\\natural_images\{}\\".format(label)
    folder_data = os.listdir(path)
    for image_path in folder_data:
        image = cv2.imread(path+image_path)
        image_resized = cv2.resize(image, (32,32))
        x_data.append(np.array(image_resized))
        y_data.append(label)
        pass
    pass

x_data = np.array(x_data)
y_data = np.array(y_data)
print('the shape of X is: ', x_data.shape, 'and that of Y is: ', y_data.shape)
x_data = x_data.astype('float32')/255


# # converting y data into categorical data

# In[47]:


y_encoded = LabelEncoder().fit_transform(y_data)
y_categorical = to_categorical(y_encoded)


# # shuffle data

# In[48]:


r = np.arange(x_data.shape[0])
np.random.seed(42)
np.random.shuffle(r)
X = x_data[r]
Y = y_categorical[r]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33)


# # create a CNN model

# In[49]:


model = models.Sequential()
model.add(layers.Conv2D(filters=32, kernel_size=(5,5), activation='relu', input_shape=X_train.shape[1:]))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dropout(rate=0.5))
model.add(layers.Dense(8, activation='softmax'))


# # compile the CNN model

# In[50]:


model.compile(
    loss='categorical_crossentropy', 
    optimizer='adam', 
    metrics=['accuracy'])


# # training model

# In[51]:


history = model.fit(X_train, Y_train, epochs=50, validation_split=0.2)


# # Plot CNN model

# In[52]:


plot_model(model)


# # Accuracy Score

# In[53]:


Y_pred = np.argmax(model.predict(X_test), axis=1)
Y_test = np.argmax(Y_test, axis = 1)
accuracy_score(Y_pred,Y_test)


# In[54]:


print(classification_report(Y_test, Y_pred))


# In[55]:


model.save('cnnmodel')

