import argparse
import cv2
import json
import os
import tqdm

def label_imgs(image_dir, output_file):
    # Get list of images
    imgs = sorted(os.listdir(image_dir))
    imgs = [os.path.join(image_dir, i) for i in imgs if i.endswith('.jpeg')]

    # Initialize labels structure.
    labels = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            labels = json.load(f)

    # Initialize counter.
    pbar = tqdm.tqdm(total=len(imgs))

    # Main labeling loop.
    for i, img_path in enumerate(imgs):
        pbar.update(1)

        # Get image id. If image already has a label, skip.
        img_id = os.path.basename(img_path).rstrip('.jpeg')
        if img_id in labels:
            continue

        # Read and show image.
        img = cv2.imread(img_path)
        cv2.imshow('images', img)

        # Get label.
        label = -1
        k = cv2.waitKey()
        if k == 27:  # ESC
            break
        elif k == ord('1'):
            label = 0
        elif k == ord('2'):
            label = 1
        labels[img_id] = label

        # Periodically dump labels. 
        if i % 30 == 0 and i > 0:
            with open(output_file, 'w') as f:
                json.dump(labels, f)

    # Final label dump.
    with open(output_file, 'w') as f:
        json.dump(labels, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Label images.')
    parser.add_argument('image_dir', help='Directory with images to label.')
    parser.add_argument(
        '--output_file', '-o', required=True,
        help='Path where the label file will be written.'
    )
    args = parser.parse_args()
    label_imgs(args.image_dir, args.output_file)