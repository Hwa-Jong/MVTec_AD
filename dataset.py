import os
import numpy as np
import cv2
import torch
import torch.utils.data as data

from easydict import EasyDict

def main():
    # dataset = Mvtec('.\\dataset\\mvtec_anomaly_detection\\capsule', mode='train', aug=[])
    dataset = Mvtec('.\\dataset\\mvtec_anomaly_detection\\capsule', mode='test', aug=[])
    x, y = dataset.__getitem__(45)
    print('finish')

class Mvtec(data.Dataset):
    def __init__(self, path, mode='train', aug=[]):
        super(Mvtec, self).__init__()
        self.path = path
        self.mode = mode
        self.aug = aug
        self.img_path = []
        self.get_data()
            
    def get_data(self):
        path = os.path.join(self.path, self.mode)
        classes = sorted(os.listdir(path))
        for cls in classes:
            p = os.path.join(path, cls)
            names = sorted(os.listdir(p))
            self.img_path += [os.path.join(p, name) for name in names]

    def augmentation(self, img):
        if 'hflip' in self.aug:
            pass
        if 'vflip' in self.aug:
            pass

        return img
        

    def __getitem__(self, index):
        path = self.img_path[index]
        x = cv2.imread(path)
        x = cv2.resize(x, (512,512))
        
        path = path.replace('test', 'ground_truth')
        path = path.replace('.png', '_mask.png')

        if os.path.exists(path):
            # y = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            # y = cv2.resize(y, (512,512))
            # y = y[..., np.newaxis]     
            
            y = torch.ones((1,), dtype=torch.int32)
        else:
            y = torch.zeros((1,), dtype=torch.int32)

        x = self.preprocessing(x)
        x = torch.from_numpy(x)

        return x, y

    def preprocessing(self, img):
        img = np.swapaxes(img, 2, 1)
        img = np.swapaxes(img, 1, 0)
        img = img.astype(np.float32)
        img /= 255.0
        return img
    

    def __len__(self):
        return len(self.img_path)

if __name__ =='__main__':
    main()

