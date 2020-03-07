import cv2
import json
import numpy as np
import os
import torch

class IllustrationsDataset(torch.utils.data.Dataset):
    """Illustrations Dataset."""

    def __init__(self, img_dir, label_file=None, augment=False):
        self.augment = augment

        # Load labels.
        if label_file is not None:
            with open(label_file) as f:
                labels = json.load(f)
            self.img_ids = list(labels.keys())
            self.labels = [labels[i] for i in self.img_ids]
        else:
            self.img_ids = [i.rstrip('.jpeg') for i in os.listdir(img_dir) if i.endswith('.jpeg')]
            self.labels = [-1 for i in self.img_ids]

        # Load images.
        imgs = [os.path.join(img_dir, '{}.jpeg'.format(i)) for i in self.img_ids]
        imgs = [cv2.imread(i) for i in imgs]
        #imgs = [cv2.resize(i, (64, 64)) for i in imgs]
        imgs = [cv2.resize(i, (128, 128))[:, :, ::-1] for i in imgs]
        imgs = [np.transpose(i, (2, 0, 1)).astype(np.float32) / 255.0 for i in imgs]
        self.imgs = imgs
        
        assert len(self.labels) == len(self.imgs)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = self.imgs[idx]
        axis_to_flip = []
        if self.augment:
            if np.random.rand() > 0.5:
                axis_to_flip.append(1)
            if np.random.rand() > 0.5:
                axis_to_flip.append(2)
            img = np.flip(img, axis_to_flip).copy()
        #return img, np.float32(self.labels[idx]), self.img_ids[idx]
        return img, self.labels[idx], self.img_ids[idx]


class Classifier(torch.nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 32, 5)
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(32, 64, 5)
        self.conv3 = torch.nn.Conv2d(64, 64, 5)
        #self.fc1 = torch.nn.Linear(32 * 11 * 11, 32)
        self.fc1 = torch.nn.Linear(64 * 12 * 12, 128)
        self.fc2 = torch.nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.nn.functional.relu(self.conv1(x)))
        x = self.pool(torch.nn.functional.relu(self.conv2(x)))
        x = self.pool(torch.nn.functional.relu(self.conv3(x)))
        #x = x.view(-1, 32 * 11 * 11)
        x = x.view(-1, 64 * 12 * 12)
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        x = x.view(-1, 2)
        return x


def get_adversarial_example(model, input_img, targets, loss_fn, eps=0.01):
    # Initialize adversarial example.
    adv_image = input_img
    for i in range(10):
        # Request the image gradients.
        adv_image = adv_image.clone().detach().requires_grad_(True)
        
        # Compute gradient.
        output = model(adv_image)
        loss = loss_fn(output, targets)
        model.zero_grad()
        loss.backward()
        grad = adv_image.grad.data

        # Do the adversarial example update and reprojection.
        adv_image = adv_image + torch.sign(grad)
        projected_diff = torch.clamp(adv_image - input_img, -eps, eps)
        adv_image = input_img + projected_diff

        # Clamp to 0-1 range.
        adv_image = torch.clamp(adv_image, 0, 1)

    # Zero-out model's gradient.
    model.zero_grad()
    return adv_image.detach()