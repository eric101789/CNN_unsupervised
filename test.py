import pandas as pd
import tensorflow as tf

# 定義參數
batch_size = 32

# 資料預處理
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1. / 255)

# 匯入測試集圖片位址csv檔案轉為DataFrame
df = pd.read_csv('test_image_paths.csv')

# 測試集
test_generator = test_datagen.flow_from_dataframe(
    dataframe=df,
    x_col='path',
    y_col='state',
    target_size=(64, 64),
    batch_size=batch_size,
    class_mode='binary')

model = tf.keras.models.load_model("model/train_model_LSTM_1.h5")
model.summary()

# test_output = model.predict(test_generator)
test_eval = model.evaluate(test_generator, verbose=1)
