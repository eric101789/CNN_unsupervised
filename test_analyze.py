import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# matplotlib.use('TkAgg')

# 讀取預測結果CSV檔案
# df1 = pd.read_csv('result/testing/test_results_1.csv')
#
# # 繪製直方圖
# plt.hist(df1['class_probability'], bins=10)
# plt.xlabel('Probability')
# plt.ylabel('Frame')
# plt.title('Probability Distribution of Test Results')
# plt.savefig('result/testing/test_probability_hist_1.png')
# plt.show()
#
#
# # 繪製折線圖
# plt.plot(df1['class_probability'])
# # 設置標籤和標題
# plt.xlabel('Sample')
# plt.ylabel('Probability')
# plt.title('Prediction Probability')
# plt.savefig('result/testing/test_probability_plot_1.png')
# plt.show()

df2 = pd.read_csv('result/testing/test_LSTM_results_epoch400.csv')

plt.figure(dpi=300)
plt.hist(df2['class_probability'], bins=10)
plt.xlabel('Probability')
plt.ylabel('Frame')
plt.title('Probability Distribution of Test LSTM Results')
plt.savefig('result/testing/test_LSTM_probability_epoch400.png')
plt.show()


# 繪製折線圖
plt.figure(dpi=300, figsize=(70, 6.5))
plt.plot(df2['class_probability'])
# 設置標籤和標題
plt.xlabel('Sample')
plt.ylabel('Probability')
plt.title('Prediction Probability')
plt.savefig('result/testing/test_LSTM_probability_plot_epoch400.png')
plt.show()

