import tensorflow as tf
from keras import Sequential
from keras.layers import MaxPooling2D, Convolution2D, Flatten, Reshape, LSTM, Dense

# 設定圖像大小和批量大小
image_size = (64, 64, 3)
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
# classifier.add(Reshape((1, -1)))

# Add a LSTM Layer
# classifier.add(LSTM(units=128))

# Step 4 - Full connection
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# Compiling the CNN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(train_generator,
               epochs=40,
               steps_per_epoch=int(10500 // batch_size),
               validation_data=val_generator,
               validation_steps=int(3000 / batch_size))

classifier.save('model/train_model_1.h5')
