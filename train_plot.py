import matplotlib.pyplot as plt

# 設定圖片和標題的列表
image1 = 'result/training/LSTM/loss_epoch100.png'
image2 = 'result/training/LSTM/loss_epoch200.png'
image3 = 'result/training/LSTM/loss_epoch300.png'
image4 = 'result/training/LSTM/loss_epoch400.png'
image5 = 'result/training/LSTM/loss_epoch500.png'
image6 = 'result/training/LSTM/loss_epoch600.png'
image7 = 'result/training/LSTM/loss_epoch700.png'
image8 = 'result/training/LSTM/loss_epoch800.png'
images = [image1, image2, image3, image4, image5, image6, image7, image8]
titles = ['Epoch=100', 'Epoch=200', 'Epoch=300', 'Epoch=400', 'Epoch=500', 'Epoch=600', 'Epoch=700', 'Epoch=800']

plt.figure(dpi=1000)
# 創建一個2x4的子圖表格
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

# 將圖片和標題逐一添加到子圖中
for i, ax in enumerate(axes.flat):
    # 讀取圖片並顯示
    img = plt.imread(images[i])
    ax.imshow(img)
    # 設定標題
    ax.set_title(titles[i])
    # 隱藏軸刻度
    ax.axis('off')

# 調整子圖之間的間距
plt.tight_layout()

plt.savefig('result/training/LSTM/loss.png')
# 顯示圖片
plt.show()
