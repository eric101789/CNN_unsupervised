import tensorflow as tf
from keras import Sequential
from keras.layers import MaxPooling2D, Convolution2D, Flatten, Reshape, LSTM, Dense
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt

# 設定圖像大小和批量大小
image_size = (64, 64, 1)
batch_size = 32

# 讀取訓練集圖像
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
train_generator = train_datagen.flow_from_directory(
    'dataset/train_set',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='binary')

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    'dataset/val_set',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='binary')

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
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

# Create a TensorBoard callback to log training metrics
# tb_callback = TensorBoard(log_dir='./logs', histogram_freq=1)

history = classifier.fit(train_generator,
                         epochs=1000,
                         steps_per_epoch=int(10500 // batch_size),
                         validation_data=val_generator,
                         validation_steps=int(3000 / batch_size),
                         # callbacks=[tb_callback]
                         )

# classifier.save('model/train_model_1.h5')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('result/training/non_LSTM/loss.png')
plt.savefig('result/training/LSTM/loss_epoch1000.png')
plt.show()


# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.savefig('result/training/non_LSTM/acc.png')
plt.savefig('result/training/LSTM/acc_epoch1000.png')
plt.show()

