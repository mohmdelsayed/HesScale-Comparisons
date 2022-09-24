import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from deepobs.pytorch.datasets.dataset import DataSet
from torch.utils.data.sampler import SubsetRandomSampler
from random import shuffle

DATA_DIR = "/datashare/ImageNet/ILSVRC2012/"
class ImageNet(DataSet):
    def __init__(self, batch_size, train_eval_size=100000):
        self._name = "imagenet"
        self._train_eval_size = train_eval_size
        super(ImageNet, self).__init__(batch_size)

    def _make_train_and_valid_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
            ])
        train_set = ImageFolder(DATA_DIR+"train/", transform=transform)
        valid_set = ImageFolder(DATA_DIR+"train/", transform=transform)
        train_loader, valid_loader = self._make_train_and_valid_dataloader_helper(train_set, valid_set)        
        return train_loader, valid_loader

    def _make_test_dataloader(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
            transforms.Resize((224, 224)),
            ])
        test_set = ImageFolder(DATA_DIR+"validation/", transform=transform)
        return self._make_dataloader(test_set)