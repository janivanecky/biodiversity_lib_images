import argparse

def extract_illustrations(page_file, output_file):
    # Read page ids for all the illustrations.
    illustration_ids = []
    with open(page_file, encoding='utf8') as f:
        l = next(f)  # Skip the table header.
        for line in f:
            line = line.split('\t')
            page_type, page_id = line[8], line[0]
            if page_type != 'Illustration':
                continue
            illustration_ids.append(page_id)

    # Remove duplicates and sort.
    illustration_ids = sorted(list(set(illustration_ids)))

    # Write to output file.
    with open(output_file, 'w') as f:
        f.write('\n'.join(illustration_ids))

    # Print status.
    print("Saved {} illustration ids to {}.".format(len(illustration_ids), output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract illustration ids from page file.')
    parser.add_argument('page_file', help='Path to page file.')
    parser.add_argument(
        '--output_file', '-o', required=True,
        help='Path where to output illustrations ids file.'
    )
    args = parser.parse_args()
    extract_illustrations(args.page_file, args.output_file)