import argparse
import cv2
import json
import numpy as np
import os
import shutil
import torch

from common import IllustrationsDataset
from common import Classifier

DEVICE = 'cpu'

def infer(image_dir, model, output_dir):
    # Set up data loading.
    dataset = IllustrationsDataset(image_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=0
    )

    # Set up model.
    device = torch.device(DEVICE)
    classifier = Classifier().to(device)
    classifier.load_state_dict(torch.load(model, map_location=DEVICE))
    
    print("Setup finished, starting inference...")

    def mkdir_p(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass

    mkdir_p(output_dir)
    mkdir_p(os.path.join(output_dir, '0'))
    mkdir_p(os.path.join(output_dir, '1'))

    # Go thrrough the data.
    for img_batch, _, ids_batch in dataloader:
        
        output = classifier(img_batch.to(device)).round().detach().numpy()

        for image_id, pred in zip(ids_batch, output):
            img_path = os.path.join(image_dir, '{}.jpeg'.format(image_id))
            output_img_path = os.path.join(output_dir, str(int(pred)), '{}.jpeg'.format(image_id))
            shutil.copyfile(img_path, output_img_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference for binary classifier.')
    parser.add_argument('--image_dir', help='Directory with images.')
    parser.add_argument('--model', required=True, help='Classifier model file.')
    parser.add_argument(
        '--output_dir', '-o', required=True,
        help='Directory where predictions will be stored.'
    )
    args = parser.parse_args()
    infer(args.image_dir, args.model, args.output_dir)