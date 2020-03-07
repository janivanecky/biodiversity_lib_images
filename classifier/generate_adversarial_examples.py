import argparse
import cv2
import json
import numpy as np
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from common import IllustrationsDataset
from common import Classifier
from common import get_adversarial_example

EPS = 0.01
DEVICE = 'cpu'

def generate_adversarial_samples(image_dir, label_file, model_path):
    # Set up data loading.
    dataset = IllustrationsDataset(image_dir, label_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=0)

    # Set up model.
    device = torch.device(DEVICE)
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(model_path, map_location=DEVICE))

    # Set up optimizer.
    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.BCELoss()

    print("Setup finished, starting training...")

    for img_batch, label_batch, img_ids in dataloader:
        # Move data to device used.
        img_batch = img_batch.to(device)
        label_batch = label_batch.to(device)

        # Get adversarial example.
        adv_img_batch = get_adversarial_example(classifier, img_batch, label_batch, loss_fn).detach()

        for img, adv_img, img_id in zip(img_batch, adv_img_batch, img_ids):
            pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train binary classifier.')
    parser.add_argument('--image_dir', help='Directory with training images.')
    parser.add_argument('--labels', required=True, help='Labels file.')
    parser.add_argument(
        '--output', '-o', required=True,
        help='Path where the trained model will be stored.'
    )
    parser.add_argument('--model', help="Path to pretrained model.")
    args = parser.parse_args()
    generate_adversarial_samples(args.image_dir, args.labels, args.model)
