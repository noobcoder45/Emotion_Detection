import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

cnn = tf.keras.models.Sequential()
cnn.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=[48, 48, 1]))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(BatchNormalization())
cnn.add(Conv2D(32, kernel_size=3, activation='relu'))
cnn.add(MaxPooling2D(pool_size=2, strides=2))
cnn.add(BatchNormalization())
cnn.add(Flatten())
cnn.add(Dense(units=256, activation='relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(units=7, activation='sigmoid'))
cnn.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])