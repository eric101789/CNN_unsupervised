import tensorflow as tf
import pandas as pd
from keras import Sequential
# from keras.layers import MaxPooling2D, Flatten, Reshape, LSTM, Dense, Convolution2D, Conv2D

from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Reshape, LSTM, Dense
from matplotlib import pyplot as plt

# from keras import Sequential
# from keras.layers import Convolution2D, MaxPooling2D, Flatten, Reshape, LSTM, Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(32, (3, 3), input_shape=(512, 512, 1), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Reshaping the output of the previous layer to a 3D tensor
classifier.add(Reshape((1, -1)))

# Add a LSTM Layer
classifier.add(LSTM(units=128))

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

batch_size = 32

# 資料預處理
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# 匯入測試集圖片位址csv檔案轉為DataFrame
df = pd.read_csv('train_image_paths_512.csv')

# 訓練集
train_generator = train_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='path',
    y_col='state',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='binary'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

df1 = pd.read_csv('val_image_paths_512.csv')

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df1,
    x_col='path',
    y_col='state',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='binary'
)

history = classifier.fit(train_generator,
                         epochs=200,
                         steps_per_epoch=int(10500 // batch_size),
                         validation_data=val_generator,
                         validation_steps=int(1000 / batch_size))

# classifier.save('model/train_model_1.h5')
classifier.save('model/1D_test/train_model_LSTM_epoch200')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('result/training/non_LSTM/loss.png')
plt.savefig('result/training/LSTM/1D_test/loss_epoch200.png')
plt.show()


# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.savefig('result/training/non_LSTM/acc.png')
plt.savefig('result/training/LSTM/1D_test/acc_epoch200.png')
plt.show()

