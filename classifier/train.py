import argparse
import cv2
import json
import numpy as np
import os
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from common import get_adversarial_example
from common import IllustrationsDataset
from common import Classifier

EPS = 0.01
DEVICE = 'cuda'
#DEVICE = 'cpu'

def train_binary_classifier(
    image_dir, label_file, output_file, epochs, tensorboard_path=None, pretrained_model=None
):
    # Set up data loading.
    train_dataset = IllustrationsDataset(image_dir, label_file, augment=True)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=0
    )
    eval_dataset = IllustrationsDataset(image_dir, label_file)
    eval_dataloader = torch.utils.data.DataLoader(
        eval_dataset, batch_size=64, num_workers=0
    )

    # Set up model.
    device = torch.device(DEVICE)
    classifier = Classifier().to(device)
    #classifier = torchvision.models.AlexNet(num_classes=2).to(device)
    if pretrained_model:
        classifier.load_state_dict(torch.load(pretrained_model, map_location=DEVICE))

    # Set up optimizer.
    loss_fn = torch.nn.CrossEntropyLoss()
    #loss_fn = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)

    print("Setup finished, starting training...")

    if tensorboard_path:
        writer = SummaryWriter(tensorboard_path)

    for e in range(epochs):
        # Training loop.
        losses = []
        for img_batch, label_batch, _ in train_dataloader:
            # Move data to device used.
            img_batch = img_batch.to(device)
            original_batch = img_batch
            label_batch = label_batch.to(device)

            # Get adversarial example.
            img_batch = get_adversarial_example(classifier, img_batch, label_batch, loss_fn, EPS).detach()

            # Training step.
            optimizer.zero_grad()
            output = classifier(img_batch)
            loss = loss_fn(output, label_batch)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())

        train_loss = np.mean(losses)
        print("Train Loss {}".format(loss))

        # Tensorboard update every epoch.
        if writer:
            writer.add_image("input", img_batch[0], e)
            writer.add_image("original_input", original_batch[0], e)

        # Infer on the train set.
        preds, labels, losses = [], [], []
        for img_batch, label_batch, _ in eval_dataloader:
            optimizer.zero_grad()
            output = classifier(img_batch.to(device))
            preds.append(output.cpu().round().detach().numpy())
            labels.append(label_batch.cpu().detach().numpy())
            loss = loss_fn(output, label_batch.to(device)).cpu().detach().numpy()
            losses.append(loss)

        # Compute accuracy.
        preds = np.argmax(np.concatenate(preds), axis=1)
        #preds = np.concatenate(preds)
        labels = np.concatenate(labels)
        tps = np.sum(np.equal(preds, labels))
        total_count = len(preds)
        accuracy = float(tps) / float(total_count)

        val_loss = np.mean(losses)
        if writer:
            writer.add_scalar("accuracy", accuracy, e)
            writer.add_scalar("val_loss", val_loss, e)
            writer.add_scalar("train_loss", train_loss, e)
        print("Epoch {}: Accuracy {}, Loss {}".format(e, accuracy, val_loss))

    print("Saving model to {}.".format(output_file))
    torch.save(classifier.state_dict(), output_file)
    if writer:
        writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train binary classifier.')
    parser.add_argument('--image_dir', help='Directory with training images.')
    parser.add_argument('--labels', required=True, help='Labels file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs.')
    parser.add_argument(
        '--output', '-o', required=True,
        help='Path where the trained model will be stored.'
    )
    parser.add_argument('--pretrained_model', help="Path to pretrained model.")
    parser.add_argument(
        '--tensorboard', help="Path to directory where tensorboard files will be stored."
    )
    args = parser.parse_args()
    train_binary_classifier(
        args.image_dir, args.labels, args.output, args.epochs, args.tensorboard, args.pretrained_model
    )
