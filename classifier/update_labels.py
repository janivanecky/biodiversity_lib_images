import argparse
import json
from copy import deepcopy

def update_labels(input_label_file, label_update_file, output_file):
    # Load original labels.
    with open(input_label_file) as f:
        original_labels = json.load(f)
    labels = deepcopy(original_labels)
    
    # Load label updates.
    with open(label_update_file) as f:
        label_updates = json.load(f)

    # Update labels.
    for label_update in label_updates:
        assert len(label_update) == 2, "Incorrect update label format."
        img_id, label = label_update
        labels[img_id] = label

    # Write new labels.
    with open(output_file, 'w') as f:
        json.dump(labels, f)

    # Print out stats.
    print("Original label count: {}".format(len(original_labels)))
    print("New label count: {}".format(len(labels)))
    overwrite_count = len(label_updates) - (len(labels) - len(original_labels))
    print("Overwrite count: {}".format(overwrite_count))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Update labels.')
    parser.add_argument('--input_labels', required=True, help='Label file to update.')
    parser.add_argument('--label_update', required=True, help='File with label updates.')
    parser.add_argument(
        '--output_file', '-o', required=True,
        help='Path where the updated label file will be written.'
    )
    args = parser.parse_args()
    update_labels(args.input_labels, args.label_update, args.output_file)