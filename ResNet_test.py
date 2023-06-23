import tensorflow as tf
from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Dense
import pandas as pd
from matplotlib import pyplot as plt


def resnet_block(inputs, filters, kernel_size):
    # First convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Second convolutional layer
    x = Conv2D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)

    # Adding the residual connection
    x = Add()([x, inputs])
    x = ReLU()(x)
    return x


def create_resnet(input_shape, num_classes):
    inputs = Input(shape=input_shape)

    # Initial convolutional layer
    x = Conv2D(32, (3, 3), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    x = resnet_block(x, 32, (3, 3))
    x = resnet_block(x, 32, (3, 3))
    x = resnet_block(x, 32, (3, 3))

    # Global average pooling
    x = GlobalAveragePooling2D()(x)

    # Fully connected layer
    x = Dense(128, activation='relu')(x)

    # Output layer
    outputs = Dense(num_classes, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


# Create the ResNet model
input_shape = (512, 512, 1)
num_classes = 1  # Example number of classes
model = create_resnet(input_shape, num_classes)

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

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
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary'
)

val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

df1 = pd.read_csv('val_image_paths_512.csv')

val_generator = val_datagen.flow_from_dataframe(
    dataframe=df1,
    x_col='path',
    y_col='state',
    target_size=(512, 512),
    batch_size=batch_size,
    class_mode='binary'
)

history = model.fit(train_generator,
                    epochs=10,
                    steps_per_epoch=int(10500 // batch_size),
                    validation_data=val_generator,
                    validation_steps=int(1000 / batch_size))

# classifier.save('model/train_model_1.h5')
model.save('model/ResNet/train_model_LSTM_epoch10')

# Plot training and validation loss over epochs
plt.plot(history.history['loss'], label='training_loss')
plt.plot(history.history['val_loss'], label='validation_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
# plt.savefig('result/training/non_LSTM/loss.png')
plt.savefig('result/ResNet/training/Loss_epoch10.png')
plt.show()

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='training_accuracy')
plt.plot(history.history['val_accuracy'], label='validation_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
# plt.savefig('result/training/non_LSTM/acc.png')
plt.savefig('result/ResNet/training/acc_epoch10.png')
plt.show()
