import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import distance_transform_edt
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from skimage import io, transform
from skimage.color import rgb2gray


class ProjectionsDataset(Dataset):
    def __init__(self, name, root_dir, size, transform=transforms.ToTensor()):
        self.root_dir = root_dir
        self.transform = transform
        self.name = name
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        
        while True:
            if torch.is_tensor(idx):
                idx = idx.tolist()

            img_base_name = os.path.join(self.root_dir, "images", self.name, "image{:04d}".format(idx))
            label_name = os.path.join(self.root_dir, "labels", self.name, "{:04d}.npy".format(idx))

            if not os.path.isfile(label_name):
                idx+=1
                continue

            label = np.load(label_name)
            label = label[:,:,:-1]

            img1_name = f"{img_base_name}a.png"
            img2_name = f"{img_base_name}b.png"
            img3_name = f"{img_base_name}c.png"

            images = list(map(
                lambda el:
                self.transform(
                    distance_transform_edt(
                        io.imread(el)[:, :, 0]
                    )
                ),
                (img1_name, img2_name, img3_name)
            ))

            channels = torch.vstack(list(images))

            sample = {'images': channels, 'label': label}
            return channels, label


def load_train(batch_size):
    return __load(batch_size, "train", 1000, shuffle=True)


def load_test(batch_size):
    return __load(batch_size, "test", 100, shuffle=False)


def __load(batch_size, name, size, shuffle):
    data = ProjectionsDataset(name, f"data/{name}", size=size)
    dl = DataLoader(data, batch_size=batch_size, shuffle=shuffle)
    return data, dl


def __show_images(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15,15))
    for i, sample in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(sample[0].squeeze().cpu().numpy().T)
    plt.show()


if __name__ == "__main__":
    data_train, loader_train = load_train(100)
    __show_images(data_train)
