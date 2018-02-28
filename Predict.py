from PIL import Image
from GetData import DataProcess
from Unet import Unet
import csv
import numpy as np


class DataPredict(object):
    def testPredict(self):
        myData = DataProcess()
        train_npy, label_npy, test_npy = myData.loadData()
        print("数据加载完成")
        myUnet = Unet()
        model = myUnet.createUnet()
        print("Unet网络创建完成")
        model.load_weights(".\\Model\\unet.h5")
        print("权值加载完成")
        predict_npy = model.predict(test_npy, batch_size=1, verbose=0)
        np.save(".\\NpyData\\predict.npy", predict_npy)
        return predict_npy

    def npy2RGB(self):
        predict_npy = self.testPredict()
        print("开始保存预测图像")
        for i in range(predict_npy.shape[0]):
            img = np.array(predict_npy[i])
            # img.shape是(128,128,1)
            self.writeCsv(i, img[:, :, 0])
            self.createImg(i, img[:, :, 0])
        print("预测图像完成")

    def writeCsv(self, i, img):
        with open(".\\PredictCsv\\%d.csv" % (i), "w", newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(img)

    def createImg(self, i, img):
        img[img > 0.5] = 255
        img[img <= 0.5] = 0
        pic = Image.fromarray(np.uint8(img))
        print("生成第" + str(i + 1) + "张图像")
        pic.save(".\\PredictImg\\%d.tif" % i)


if __name__ == '__main__':
    myPredict = DataPredict()
    myPredict.npy2RGB()
