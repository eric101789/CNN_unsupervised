import csv
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

# 創建CSV文件，並添加表頭
csvfile = open('model/training_logs.csv', 'w', newline='')
fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# 訓練模型並將結果寫入CSV文件
for epoch in range(40):
    history = classifier.fit(train_generator,
                             epochs=1,
                             steps_per_epoch=int(10500 // batch_size),
                             validation_data=val_generator,
                             validation_steps=int(3000 / batch_size))

    # 將訓練和驗證損失、精度寫入CSV文件
    writer.writerow({'epoch': epoch + 1,
                     'train_loss': history.history['loss'][0],
                     'train_accuracy': history.history['accuracy'][0],
                     'val_loss': history.history['val_loss'][0],
                     'val_accuracy': history.history['val_accuracy'][0]})

# 關閉CSV文件
csvfile.close()

# 保存模型
# classifier.save('model/train_model_1.h5')
