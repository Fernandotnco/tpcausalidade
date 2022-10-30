import cv2
import torch.utils.data as data
from torchvision import transforms
import os
import random
import matplotlib.pyplot as plt

def CreateBinaryDataset(base_dir, yes_dirs, no_dirs, dataset_dir, val = 0.15):
    train_labels = []
    train_images = []
    val_labels = []
    val_images = []
    for f in os.listdir(base_dir):
        if f in yes_dirs:
            for img in os.listdir(os.path.join(base_dir, f)):
                num = random.random()
                img = os.path.join(f, img)
                if(num < val):
                    val_images.append(img)
                    val_labels.append(1)
                else:
                    train_images.append(img)
                    train_labels.append(1)

        if f in no_dirs:
            for img in os.listdir(os.path.join(base_dir, f)):
                img = os.path.join(f, img)
                num = random.random()
                if(num < val):
                    val_images.append(img)
                    val_labels.append(0)
                else:
                    train_images.append(img)
                    train_labels.append(0)

    train_dataset = ImageDataset(train_images, train_labels, dataset_dir)
    val_dataset = ImageDataset(val_images, val_labels, dataset_dir)

    return train_dataset, val_dataset

class ImageDataset(data.TensorDataset):
    def __init__(self, img_list, labels_list, dir, test = False):
        self.img_list = img_list
        self.labels = labels_list
        self.dir = dir
        self.test = test

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        '''
        get tensor type preprocessed Image
        '''
        convert_tensor = transforms.ToTensor()
        img = cv2.imread(os.path.join(self.dir, self.img_list[index]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img,(256,256))
        img = convert_tensor(img)
        return img, self.labels[index]

if __name__ == "__main__":
    os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

    base_dir = 'C:/Users/ferna/Documents/UFMG/8 Semestre/Causalidade/tpcausalidade/dataset'
    train, val = CreateBinaryDataset(base_dir, ['yes'], ['no', 'pituary', 'meningioma'], base_dir)

    train_dataloader = data.DataLoader(train, batch_size = 8, shuffle=True, num_workers=0, drop_last = False)

    for img, label in train_dataloader:
        print(label)
        plt.imshow(img[0][0])
        plt.show()