import os
import torch
import numpy as np
import matplotlib.pyplot as plt


from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from skimage import io, transform
from skimage.color import rgb2gray


class FaceLandmarksDataset(Dataset):
    """Face Landmarks dataset."""

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

            img1_name = f"{img_base_name}a.png"
            img2_name = f"{img_base_name}b.png"
            img3_name = f"{img_base_name}c.png"

            image1 = io.imread(img1_name)[:, :, 0]
            image2 = io.imread(img2_name)[:, :, 0]
            image3 = io.imread(img3_name)[:, :, 0]

            image1 = self.transform(image1)
            image2 = self.transform(image2)
            image3 = self.transform(image3)

            sample = {'projection_a': image1,"projection_b": image2, "projection_c": image3, 'label': label}
            return sample 


def load_data(batch_size):

    data_train = FaceLandmarksDataset("train", "data/train", size=1000)
    dl_train = DataLoader(data_train, batch_size=batch_size, shuffle=True)

    return (data_train, dl_train)


def show_images(dataset, num_samples=20, cols=4):
    plt.figure(figsize=(15,15))
    for i, sample in enumerate(dataset):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.imshow(sample["projection_c"].squeeze().cpu().numpy())
    plt.show()

if __name__ == "__main__":
    data_train, loader_train = load_data(100)
    show_images(data_train)
