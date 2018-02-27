from GetData import DataProcess
from Unet import Unet
from keras.preprocessing import image
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
        np.save(".\\NpyData\\predict.npy",predict_npy)
        print("开始保存预测图像")
        for i in range(predict_npy.shape[0]):
            img=predict_npy[i]
            img=image.array_to_img(img)
            print("生成第"+str(i+1)+"张图像")
            img.save(".\\PredictImg\\%d.jpg"%(i))
        print("预测图像完成")

if __name__=='__main__':
    myPredict=DataPredict()
    myPredict.testPredict()