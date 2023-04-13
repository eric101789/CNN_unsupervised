import pandas as pd
import matplotlib.pyplot as plt

# 讀取預測結果CSV檔案
df1 = pd.read_csv('result/testing/test_results.csv')

# 繪製直方圖
plt.hist(df1['class_probability'], bins=10)
plt.xlabel('Probability')
plt.ylabel('Frame')
plt.title('Probability Distribution of Test Results')
plt.show()
plt.savefig('result/testing/test_probability.png')

df2 = pd.read_csv('result/testing/test_LSTM_results.csv')

plt.hist(df2['class_probability'], bins=10)
plt.xlabel('Probability')
plt.ylabel('Frame')
plt.title('Probability Distribution of Test LSTM Results')
plt.show()
plt.savefig('result/testing/test_LSTN_probability.png')
