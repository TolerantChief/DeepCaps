'''
Dataset loading and data transformation classes.
'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torchvision
import os


class Jamones:
    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.2), scale=(0.95, 1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 28
        self.num_class = 26

    def __call__(self):

        transform = transforms.Compose([
        # shift by 2 pixels in either direction with zero padding.
        transforms.Grayscale(),
        transforms.Resize((self.img_size,self.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.RandomRotation(10),
        transforms.RandomCrop(self.img_size, padding=2),
        transforms.ToTensor(),
        #transforms.Normalize(mean, std)
    ])

        test_size = 0.2
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(self.data_path, 'JAMONES'), transform=transform)
        num_data = len(dataset)
        num_test = int(test_size * num_data)
        num_train = num_data - num_test
        train_dataset, test_dataset = torch.utils.data.random_split(dataset, [num_train, num_test])
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)


        return train_loader, test_loader, self.img_size, self.num_class

class FashionMNIST:

    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.2), scale=(0.95,1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 28
        self.num_class = 10

    def __call__(self):

        train_loader = DataLoader(datasets.FashionMNIST(root=self.data_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale
                                                        ), transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.FashionMNIST(root=self.data_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.ToTensor()),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)


        return train_loader, test_loader, self.img_size, self.num_class

class Cifar10:

    def __init__(self, data_path, batch_size, shuffle, num_workers=4, rotation_degrees=30, translate=(0,0.2), scale=(0.95, 1.2)):

        self.data_path = data_path
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.rotation = rotation_degrees
        self.translate = translate
        self.scale = scale
        self.img_size = 28
        self.num_class = 10

    def __call__(self):

        train_loader = DataLoader(datasets.CIFAR10(root=self.data_path,
                                                        train=True,
                                                        download=True,
                                                        transform=transforms.Compose([transforms.RandomAffine(
                                                                                                            degrees=self.rotation,
                                                                                                            translate=self.translate,
                                                                                                            scale=self.scale
                                                        ), transforms.Grayscale(), transforms.Resize(self.img_size), transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)

        test_loader = DataLoader(datasets.CIFAR10(root=self.data_path,
                                                        train=False,
                                                        download=True,
                                                        transform=transforms.Compose([
                                                                                      transforms.Grayscale(),
                                                                                      transforms.Resize(self.img_size),
                                                                                      transforms.ToTensor()])),
                                                        batch_size=self.batch_size,
                                                        shuffle=self.shuffle,
                                                        num_workers=self.num_workers)



        return train_loader, test_loader, self.img_size, self.num_class
