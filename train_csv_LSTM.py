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

# 讀取驗證集圖像
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)
val_generator = val_datagen.flow_from_directory(
    'dataset/val_set',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='binary')

# 初始化CNN神經網路
classifier = Sequential()

# 新增第一層卷積層
classifier.add(Convolution2D(32, 3, 3, input_shape=(64, 64, 3), activation='relu'))

# 新增池化層
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 新增第二層卷積層
classifier.add(Convolution2D(32, 3, 3, activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# 新增扁平層
classifier.add(Flatten())

# 調整上一層的輸出資料為3D張量
classifier.add(Reshape((1, -1)))

# 新增LSTM層
classifier.add(LSTM(units=128))

# 新增全連接層
classifier.add(Dense(units=128, activation='relu'))
classifier.add(Dense(units=1, activation='sigmoid'))

# 編譯CNN模型
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 創建CSV文件，並添加表頭
csvfile = open('result/training/LSTM/training_LSTM_logs.csv', 'w', newline='')
fieldnames = ['epoch', 'train_loss', 'train_accuracy', 'val_loss', 'val_accuracy']
writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
writer.writeheader()

# 訓練模型並將結果寫入CSV文件
for epoch in range(1000):
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
classifier.save('model/train_model_LSTM_epoch1000.h5')
