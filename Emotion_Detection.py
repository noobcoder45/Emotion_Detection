from Model import cnn
import numpy as np
import pandas as pd
import PIL.Image
from tkinter import *
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
import keras.utils as image

train_data_gen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
train_data=train_data_gen.flow_from_directory('./dataset/train',color_mode='grayscale',target_size=(48,48))
test_data_gen=ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.2,horizontal_flip=True)
test_data=test_data_gen.flow_from_directory('./dataset/test',color_mode='grayscale',target_size=(48,48))

history = cnn.fit(x=train_data, validation_data=test_data, epochs=45)

root = Tk()
root.withdraw()

file_path = askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.gif")])

img = PIL.Image.open(file_path)


test_image=image.load_img(file_path,target_size=(48,48),grayscale=True)
test_image=image.img_to_array(test_image)
test_image=np.expand_dims(test_image,axis=0)

plt.imshow(img)
plt.axis('off')
plt.show(block=False)

result=cnn.predict(test_image)
predicted_labels = np.argmax(result, axis=1)
emotions=['Angry','Disgust','Fear','Happy', 'Nuetral','Sad','Surprise']
prediction=emotions[predicted_labels[0]]

root = Tk()
label = Label(root, text=prediction)
label.pack()
root.mainloop()