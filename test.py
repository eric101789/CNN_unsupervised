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

model = tf.keras.models.load_model("model/train_model_epoch100.h5")
model.summary()

test_output = model.predict(test_generator, batch_size=batch_size, verbose=1)
# test_eval = model.evaluate(test_generator, batch_size=batch_size, verbose=1)

# 取得每個圖片的預測結果和對應的機率
predicted_classes = test_output.argmax(axis=1)
class_probabilities = test_output.max(axis=1)

# 將預測結果和機率寫入CSV文件
results_df = pd.DataFrame({'predicted_class': predicted_classes, 'class_probability': class_probabilities})
results_df.to_csv('result/testing/test_results_1.csv', index=False)

model1 = tf.keras.models.load_model("model/train_model_LSTM_epoch100.h5")
model1.summary()

test_output1 = model1.predict(test_generator, batch_size=batch_size, verbose=1)
# test_eval1 = model1.evaluate(test_generator, batch_size=batch_size, verbose=1)

# 取得每個圖片的預測結果和對應的機率
predicted_classes1 = test_output1.argmax(axis=1)
class_probabilities1 = test_output1.max(axis=1)

# 將預測結果和機率寫入CSV文件(LSTM)
results_df = pd.DataFrame({'predicted_class': predicted_classes1, 'class_probability': class_probabilities1})
results_df.to_csv('result/testing/test_LSTM_results_1.csv', index=False)
