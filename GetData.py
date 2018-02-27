import os

from PIL import Image


def globel():
    train_path = ".\\ImgData\\TrainData\\"
    label_path = ".\\ImgData\\LabelData\\"
    test_path = ".\\ImgData\\TestData\\"
    img_type = "tif"
    img_width = 128
    img_height = 128
    npy_path = ".\\NpyData\\"
    trainout_path = ".\\ImgData\\TrainDataOut\\"
    labelout_path = ".\\ImgData\\LabelDataOut\\"
    testout_path = ".\\ImgData\\TestDataOut\\"

class DataProcess(object):
    def __init__(self,train_path = ".\\ImgData\\TrainData\\",label_path = ".\\ImgData\\LabelData\\",test_path = ".\\ImgData\\TestData\\",
                 trainout_path=".\\ImgData\\TrainDataOut\\",labelout_path=".\\ImgData\\LabelDataOut\\",testout_path=".\\ImgData\\TestDataOut\\",
                 img_type = "tif",img_width = 128,img_height = 128,npy_path = ".\\NpyData\\"):
        self.train_path=train_path
        self.label_path=label_path
        self.test_path=test_path
        self.img_type=img_type
        self.img_width=img_width
        self.img_height=img_height
        self.npy_path=npy_path
        self.trainout_path=trainout_path
        self.labelout_path=labelout_path
        self.testout_path=testout_path
    def glayAndResize(self):
        trainDataList =os.listdir(self.train_path)
        labelDataList = os.listdir(self.label_path)
        testDataList = os.listdir(self.test_path)
        for i in trainDataList:
            path = self.train_path+i
            im = Image.open(path)
            step_1=im.convert('L')
            step_2=step_1.resize((self.img_width,self.img_height))
            step_2.save(self.trainout_path+i)
        for i in labelDataList:
            path = self.label_path+i
            im = Image.open(path)
            step_1=im.convert('L')
            step_2=step_1.resize((self.img_width,self.img_height))
            step_2.save(self.labelout_path+i)
        for i in testDataList:
            path = self.test_path+i
            im = Image.open(path)
            step_1=im.convert('L')
            step_2=step_1.resize((self.img_width,self.img_height))
            step_2.save(self.testout_path+i)

if __name__=='__main__':
    myData=DataProcess()
    myData.glayAndResize()