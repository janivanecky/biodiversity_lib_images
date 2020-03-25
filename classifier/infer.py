import argparse
import cv2
import json
import numpy as np
import os
import shutil
import torch
import torchvision

from common import IllustrationsDataset

DEVICE = 'cpu'

def infer(image_dir, model, output_file, output_dir):
    # Set up data loading.
    dataset = IllustrationsDataset(image_dir, stream_imgs=True)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=64, num_workers=32,
    )

    # Set up model.
    device = torch.device(DEVICE)
    classifier = torchvision.models.AlexNet(num_classes=2).to(device)
    classifier.load_state_dict(torch.load(model, map_location=DEVICE))
    
    print("Setup finished, starting inference...")

    def mkdir_p(dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass

    if output_dir:
        mkdir_p(output_dir)
        mkdir_p(os.path.join(output_dir, '0'))
        mkdir_p(os.path.join(output_dir, '1'))

    predictions = {}

    # Go through the data.
    for img_batch, _, ids_batch in dataloader:
        
        output = classifier(img_batch.to(device)).detach().numpy()

        for image_id, pred in zip(ids_batch, output):
            pred_raw = pred
            pred = np.argmax(pred)
            predictions[image_id] = [float(x) for x in list(pred_raw)]
            
            if output_dir:
                img_path = os.path.join(image_dir, '{}.jpeg'.format(image_id))
                output_img_path = os.path.join(output_dir, str(int(pred)), '{}.jpeg'.format(image_id))
                shutil.copyfile(img_path, output_img_path)


    with open(output_file, 'w') as f:
        json.dump(predictions, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run inference for binary classifier.')
    parser.add_argument('--image_dir', help='Directory with images.')
    parser.add_argument('--model', required=True, help='Classifier model file.')
    parser.add_argument('--output', required=True, help="JSON output with predictions.")
    parser.add_argument('--output_dir', help='Directory where predictions will be stored.')
    args = parser.parse_args()
    infer(args.image_dir, args.model, args.output, args.output_dir)