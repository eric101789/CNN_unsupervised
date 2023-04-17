"""
程式碼功能
    1. 將數據子載波( len = 52 )置於一個 list 的前面，然後進行歸一化 (normalization)
    2. 將數據子載波( len = 52 )補 0 直到 len = 64 ，再轉換成矩陣
    3. 將矩陣轉換成圖片並儲存，另外還可以等比例放大圖片
參考 :
    補 0 : https://blog.csdn.net/Hodors/article/details/117438511
    list 轉成矩陣 : https://www.delftstack.com/zh-tw/howto/python/list-to-matrix-python/#%e5%9c%a8-python-%e4%b8%ad%e4%bd%bf%e7%94%a8%e8%bf%b4%e5%9c%88%e5%92%8c%e5%88%97%e8%a1%a8%e5%88%87%e7%89%87%e5%b0%87%e5%88%97%e8%a1%a8%e8%bd%89%e6%8f%9b%e7%82%ba%e9%99%a3%e5%88%97%e6%88%96%e7%9f%a9%e9%99%a3
    等比例放大圖片 : https://www.zhihu.com/question/311903523
"""
import csv
import re
from math import sqrt, atan2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2

if __name__ == "__main__":
    """
    This script file demonstrates how to transform raw CSI out from the ESP32 into CSI-amplitude and CSI-phase.
    """
    FILE_NAME1 = "C:/PythonProject/plot_csi_figure/src/old/csv_raw/unmanned.csv"                    # 原始 csv 檔案
    PATH = "C:/PythonProject/plot_csi_figure/src/old/dataset/0315/normalization_test/"          # 儲存灰階圖的資料夾位置

    f1 = open(FILE_NAME1)

    loop1 = 0
    loop_n = 10      # 設定總共要輸出多少筆資料

    j0 = 7600         # 選擇要從哪一行開始讀取

    for j1, l1 in enumerate(f1.readlines()):
        if j1 >= j0:        # 因為 j1 是從 0 開始計算，如果要讀取 20 行，需要將 j1 設為  19
            imaginary1 = []
            real1 = []
            amplitudes1 = []
            phases1 = []

            # Parse string to create integer list
            csi_string1 = re.findall(r"\[(.*)\]", l1)[0]
            csi_raw1 = [int(x) for x in csi_string1.split(" ") if x != '']

            # Create list of imaginary and real numbers from CSI
            for i in range(len(csi_raw1)):
                if i % 2 == 0:
                    imaginary1.append(csi_raw1[i])
                else:
                    real1.append(csi_raw1[i])

            # Transform imaginary and real into amplitude and phase
            for i in range(int(len(csi_raw1) / 2)):
                # 把計算後的振幅值取 3 位小數點後，放入 list 裡
                amplitudes1.append(format(sqrt(imaginary1[i] ** 2 + real1[i] ** 2), '.3f'))
                phases1.append(format(atan2(imaginary1[i], real1[i]), '.3f'))

            if loop1 >= loop_n:
                break

            data_sub = []
            # 將數據子載波的上、下兩部分整合成一個新的 list
            # data_sub = [element for e in [amplitudes1[6:32], amplitudes1[33:59]] for element in e]
            data_sub = amplitudes1[6:32] + amplitudes1[33:59]
            # print('數據子載波 : ', data_sub)

            # 計算補 0 前的歸一化值
            # 計算每列的最大值和最小值
            max_vals = float(max(data_sub[0:20]))
            min_vals = float(min(data_sub[25:]))
            # print('Maximum : ', max_vals)
            # print('Minimum : ', min_vals)

            # 進行歸一化 (normalization) 計算
            arr = np.array(data_sub)
            normalized_arr = []
            for n in data_sub:
                normalized_arr.append(format((float(n) - min_vals) / (max_vals - min_vals), '.3f'))
            # print('正歸化的數據子載波 : ', normalized_arr)

            # 進行正歸化，將其轉換到 0 到 255 之間
            scale_factor = 255
            normalization_piexl = []
            for x in normalized_arr:
                normalization_piexl.append(format(float(x) * scale_factor, '.3f'))
            # print('補0前正歸化(0~255) : ', normalization_piexl)
            # print(len(normalization_piexl))

            normalization_piexl.extend(0 for _ in range(abs(64-len(data_sub))))
            # print('補0後的數據子載波 : ', normalization_piexl)

            matrix_data_sub = list(np.array_split(normalization_piexl, 8))
            matrix = np.array(matrix_data_sub)
            # print('數據子載波的矩陣 : \n', matrix)

            size = (8, 8)
            img = Image.new('L', size)
            pixels = img.load()
            for i in range(size[0]):
                for j in range(size[1]):
                    pixels[i, j] = int(float(matrix[j][i]))

            # 等比例放大圖片
            w, h = img.size
            # print("原圖片的大小 : ", img.size)
            width = img.size[0]   # 圖片寬度
            height = img.size[1]   # 圖片高度
            scale = 8.0
            reimg = img.resize((int(width * scale), int(height * scale)), Image.ANTIALIAS)
            # reimg = cv2.resize(img, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_LINEAR)

            # print("放大後圖片的大小 : ", reimg.size)

            # 繪製水平和垂直線條
            # fig, ax = plt.subplots()
            # ax.imshow(img, cmap='gray')
            # ax.imshow(reimg, cmap='gray')

            # ycoords = [i for i in range(1, 64) if i % 8 == 0]               # 64*64
            # ycoords = [x / 2 for x in range(1, 16) if x % 2 == 1]         # 8x8
            # for i in ycoords:
                # ax.axhline(i*h/8, color='red', linewidth=0.5)
                # ax.axvline(i*w/8, color='red', linewidth=0.5)

            # 儲存圖片
            address = PATH + "unmanned-{}.png".format(j1)   # 編輯圖片的名稱
            reimg.save(address)                           # 儲存圖片
            # plt.savefig(address, dpi=300)                            # 儲存圖片
            # plt.show()

            loop1 += 1

    print("已儲存圖片!")
