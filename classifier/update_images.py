import argparse
import json
import os
import shutil

def update_images(label_update_file, src_img_dir, dst_img_dir):
    # Load label updates.
    with open(label_update_file) as f:
        label_updates = json.load(f)

    # Update dst image directory.
    for label_update in label_updates:
        assert len(label_update) in {1, 2}, "Incorrect update label format."
        if len(label_update) == 2:
            img_id, _ = label_update
        else:
            img_id = label_update[0]

        # Copy over image.
        src_img_path = os.path.join(src_img_dir, '{}.jpeg'.format(img_id))
        dst_img_path = os.path.join(dst_img_dir, '{}.jpeg'.format(img_id))
        shutil.copyfile(src_img_path, dst_img_path)

    print("Copied over {} images.".format(len(label_updates)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update images.')
    parser.add_argument('--label_update', required=True, help='File with label updates.')
    parser.add_argument('--src_img_dir', required=True, help='Directory where to copy from.')
    parser.add_argument('--dst_img_dir', required=True, help='Directory where to copy to.')
    args = parser.parse_args()
    update_images(args.label_update, args.src_img_dir, args.dst_img_dir)