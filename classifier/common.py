import cv2
import json
import numpy as np
import os
import torch
from multiprocessing.pool import ThreadPool

class IllustrationsDataset(torch.utils.data.Dataset):
    """Illustrations Dataset."""

    def __init__(self, img_dir, label_file=None, augment=False, stream_imgs=False):
        self.augment = augment
        self.img_dir = img_dir
        self.stream_imgs = stream_imgs

        # Load labels.
        if label_file is not None:
            with open(label_file) as f:
                labels = json.load(f)
            
            for i in list(labels.keys()):
                if labels[i] == 2:
                    del labels[i]
            self.img_ids = list(labels.keys())
            self.labels = [labels[i] for i in self.img_ids]
        else:
            self.img_ids = [i.rstrip('.jpeg') for i in os.listdir(img_dir) if i.endswith('.jpeg')]
            self.labels = [-1 for i in self.img_ids]

        # Load images.
        self.imgs = None
        if not stream_imgs:
            imgs = [os.path.join(img_dir, '{}.jpeg'.format(i)) for i in self.img_ids]
            imgs = [cv2.imread(i) for i in imgs]
            imgs = [cv2.resize(i, (256, 256))[:, :, ::-1] for i in imgs]
            imgs = [np.transpose(i, (2, 0, 1)).astype(np.float32) / 255.0 for i in imgs]
            self.imgs = imgs
        
            assert len(self.labels) == len(self.imgs)

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        # Get image.
        if self.stream_imgs:
            img = self.img_ids[idx]
            img = os.path.join(self.img_dir, '{}.jpeg'.format(img))
            img = cv2.imread(img)
            img = cv2.resize(img, (256, 256))[:, :, ::-1]
            img = np.transpose(img, (2, 0, 1)).astype(np.float32) / 255.0
        else:
            img = self.imgs[idx]

        # Augment image.
        axis_to_flip = []
        if self.augment:
            if np.random.rand() > 0.5:
                axis_to_flip.append(1)
            if np.random.rand() > 0.5:
                axis_to_flip.append(2)
            img = np.flip(img, axis_to_flip).copy()

        return img, self.labels[idx], self.img_ids[idx]

    def summary(self):
        print("Dataset stats:")
        print("==================")
        print("{} total samples".format(len(self.img_ids)))
        if self.labels[0] >= 0:
            for i in range(2):
                class_i_samples = [l for l in self.labels if l == i]
                print("{} class {} samples".format(len(class_i_samples), i))
        print()
        
        if self.augment:
            print("Augmentation on")
        else:
            print("Augmentation off")
        print()


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