import matplotlib.pyplot as plt
import torch
import torch.utils.data as data
import torchvision
from PIL import Image
from utils.parser import get_config
import os

conf_file = os.path.join('', "./configs/alex_net.yaml")
cfg = get_config(conf_file)


def transform(image_size):
    return torchvision.transforms.Compose([
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomResizedCrop((image_size, image_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])


def trainloader(mini_batch_size, train_dir=None):
    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(train_dir, transform=transform(cfg.INPUT_IMAGE_SIZE)),
        batch_size=mini_batch_size, shuffle=True, num_workers=6
    )


def testloader(mini_batch_size, test_dir=None):
    return torch.utils.data.DataLoader(
        torchvision.datasets.ImageFolder(test_dir, transform=transform(cfg.INPUT_IMAGE_SIZE)),
        batch_size=mini_batch_size, shuffle=False, num_workers=6
    )


if __name__ == "__main__":
    image = Image.open("../data/train/2/cat.0.jpg")
    print(type(image))
    image = torchvision.transforms.RandomHorizontalFlip()(image)
    image = torchvision.transforms.RandomResizedCrop((224, 224))(image)
    image = torchvision.transforms.ToTensor()(image)
    image = torchvision.transforms.Normalize(0.5, 0.5)(image)
    print(image.size())
    plt.imshow(image.permute(1, 2, 0))
    plt.show()
    print(image.max(), image.min())
