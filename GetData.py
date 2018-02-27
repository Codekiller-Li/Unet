import fnmatch
import os
import glob
import numpy as np
import natsort
from keras.preprocessing import image
from PIL import Image

width=128
height=128

class DataProcess(object):
    def __init__(self, train_path=".\\ImgData\\TrainData\\", label_path=".\\ImgData\\LabelData\\",
                 test_path=".\\ImgData\\TestData\\", trainout_path=".\\ImgData\\TrainDataOut\\",
                 labelout_path=".\\ImgData\\LabelDataOut\\", testout_path=".\\ImgData\\TestDataOut\\",
                 img_type="tif", img_width=width, img_height=height, train_npy_path=".\\NpyData\\train.npy",
                 label_npy_path=".\\NpyData\\label.npy", test_npy_path=".\\NpyData\\test.npy"):
        self.train_path = train_path
        self.label_path = label_path
        self.test_path = test_path
        self.img_type = img_type
        self.img_width = img_width
        self.img_height = img_height
        self.trainout_path = trainout_path
        self.labelout_path = labelout_path
        self.testout_path = testout_path
        self.train_npy_path = train_npy_path
        self.label_npy_path = label_npy_path
        self.test_npy_path = test_npy_path

    def glayAndResize(self):
        trainDataList = os.listdir(self.train_path)
        labelDataList = os.listdir(self.label_path)
        testDataList = os.listdir(self.test_path)
        for i in trainDataList:
            path = self.train_path + i
            im = Image.open(path)
            step_1 = im.convert('L')
            step_2 = step_1.resize((self.img_width, self.img_height))
            step_2.save(self.trainout_path + i)
        for i in labelDataList:
            path = self.label_path + i
            im = Image.open(path)
            step_1 = im.convert('L')
            step_2 = step_1.resize((self.img_width, self.img_height))
            step_2.save(self.labelout_path + i)
        for i in testDataList:
            path = self.test_path + i
            im = Image.open(path)
            step_1 = im.convert('L')
            step_2 = step_1.resize((self.img_width, self.img_height))
            step_2.save(self.testout_path + i)
        print("图片准备完成")

    def createData(self):
        train_img = glob.glob(self.trainout_path + "*." + self.img_type)
        label_img = glob.glob(self.labelout_path + "*." + self.img_type)
        test_img = glob.glob(self.testout_path + "*." + self.img_type)
        print("Train图片数量：", len(train_img))
        print("Test图片数量：", len(test_img))
        train_npy = np.ndarray((len(train_img), self.img_height, self.img_width, 1), dtype=np.uint8)
        label_npy = np.ndarray((len(label_img), self.img_height, self.img_width, 1), dtype=np.uint8)
        test_npy = np.ndarray((len(test_img), self.img_height, self.img_width, 1), dtype=np.uint8)

        train_list = fnmatch.filter(os.listdir(self.trainout_path), "*." + self.img_type)
        train_list = natsort.natsorted(train_list)
        label_list = fnmatch.filter(os.listdir(self.labelout_path), "*." + self.img_type)
        label_list = natsort.natsorted(label_list)
        test_list = fnmatch.filter(os.listdir(self.testout_path), "*." + self.img_type)
        test_list = natsort.natsorted(test_list)
        print("开始将一通道图片存储为.npy格式文件")
        k = int(0)
        for i in train_list:
            img = image.load_img(self.trainout_path + i, grayscale=True)
            img = image.img_to_array(img)
            train_npy[k] = img
            k += 1
        np.save(self.train_npy_path, train_npy)
        k = int(0)
        for i in label_list:
            img = image.load_img(self.labelout_path + i, grayscale=True)
            img = image.img_to_array(img)
            label_npy[k] = img
            k += 1
        np.save(self.label_npy_path, label_npy)
        k = int(0)
        for i in test_list:
            img = image.load_img(self.testout_path + i, grayscale=True)
            img = image.img_to_array(img)
            test_npy[k] = img
            k += 1
        np.save(self.test_npy_path, test_npy)
        print("存储npy文件完成")

    def loadData(self):
        train_npy=np.load(self.train_npy_path).astype('float32')
        label_npy=np.load(self.label_npy_path).astype('float32')
        test_npy=np.load(self.test_npy_path).astype('float32')

        train_npy /=255
        label_npy /=255
        test_npy /=255

        label_npy[label_npy>0.5]=1
        label_npy[label_npy<=0.5]=0
        #二值化
        return train_npy,label_npy,test_npy

if __name__ == '__main__':
    myData = DataProcess()
    myData.glayAndResize()
    myData.createData()