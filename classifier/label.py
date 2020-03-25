import argparse
import cv2
import json
import os
import tqdm

def label_imgs(image_dir, output_file, relabel_dontcare=False):
    # Get list of images
    imgs = sorted(os.listdir(image_dir))
    imgs = [os.path.join(image_dir, i) for i in imgs if i.endswith('.jpeg')]

    # Initialize labels structure.
    labels = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as f:
            labels = json.load(f)

    # Get image ids.
    imgs = [(i, os.path.basename(i).rstrip('.jpeg')) for i in imgs]

    # Get images to label.
    # Skip already labeled imgs.
    # In case `relabel_dontcare` is specified, don't skip images labeled 'dontcare'.
    imgs_to_label = [
        (img_path, img_id)
        for img_path, img_id in imgs
        if (
            img_id not in labels or
            (relabel_dontcare and labels[img_id] == 2)
        )
    ]

    # Initialize counter.
    pbar = tqdm.tqdm(total=len(imgs_to_label))

    # Main labeling loop.
    new_labels = 0
    for i, (img_path, img_id) in enumerate(imgs_to_label):
        pbar.update(1)

        # Read and show image.
        img = cv2.imread(img_path)
        img = cv2.resize(img, (512, 512))
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
        elif k == ord('3'):
            label = 2
        labels[img_id] = label

        new_labels += 1

        # Periodically dump labels. 
        if new_labels % 30 == 0:
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
    parser.add_argument('--relabel_dontcare', action='store_true', help='Whether to relabel all dontcare labels.')
    args = parser.parse_args()
    label_imgs(args.image_dir, args.output_file, args.relabel_dontcare)