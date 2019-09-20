import torch
import numpy as np
import torchvision

class RoadSceneDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images,
            offsets,
            angles,
            transform=torchvision.transforms.ToTensor(),
    ):
        # each image is stored as np array (32,32) [0,255], the dataset is stored as numpy array (N, 32, 32), (N), (N)
        self.images = images.astype(np.uint8)
        self.offsets = offsets.astype(np.float32)
        self.angles = angles.astype(np.float32)
        self.transform = transform # Totensor will transform (H,W,C) np array [0,255] into (C, H, W) tensor [0,1.0]
        assert self.images.shape[0] == self.offsets.shape[0], "Row mismatch"
        assert self.offsets.shape[0] == self.angles.shape[0], "Row mismatch"
        self.num_obs = self.offsets.shape[0]

    def __len__(self):
        return self.num_obs

    def __getitem__(self, ix):
        image = self.images[ix]
        offset = self.offsets[ix]
        angle = self.angles[ix]
        if self.transform:
            image = self.transform(image)
        return image, offset, angle


class LidarDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            images,
            labels,
            transform=torchvision.transforms.ToTensor(),
    ):
        # each image is stored as np array (64,64), the dataset is stored as numpy array (N, 64, 64), (N)
        self.images = images.astype(np.float32)
        self.labels = labels
        self.transform = transform # Totensor will transform (H,W,C) np array into (C, H, W) tensor
        assert self.images.shape[0] == self.labels.shape[0], "Row mismatch"
        self.num_obs = self.labels.shape[0]

    def __len__(self):
        return self.num_obs

    def __getitem__(self, ix):
        image = self.images[ix]
        label = self.labels[ix]
        if self.transform:
            image = self.transform(image)
        return image, label
