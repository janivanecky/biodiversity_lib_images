import argparse

def extract_illustrations(page_file, output_file):
    # Read page ids for all the illustrations and the other pages.
    illustration_ids = set()
    non_illustration_ids = set()

    with open(page_file, encoding='utf8') as f:
        next(f)  # Skip the table header.
        for line in f:
            line = line.split('\t')
            page_type, page_id, page_prefix = line[8], line[0], line[6]
            is_correct_page_type = page_type == 'Drawing' or page_type == 'Illustration'
            is_correct_prefix = (
                'plate' in page_prefix.lower()
                or 'fig' in page_prefix.lower()
                or 'pl.' in page_prefix.lower()
                or 'pl ' in page_prefix.lower()
            )
            if is_correct_page_type and is_correct_prefix:
                illustration_ids.add(page_id)
            else:
                non_illustration_ids.add(page_id)
    # We want to keep only page which are exclusively illustrations.
    illustration_ids = illustration_ids - non_illustration_ids
    illustration_ids = sorted(list(illustration_ids))

    # Write to output file.
    with open(output_file, 'w') as f:
        f.write('\n'.join(illustration_ids))

    # Print status.
    print("Saved {} page ids to {}.".format(len(illustration_ids), output_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract illustration ids from page file.')
    parser.add_argument('page_file', help='Path to page file.')
    parser.add_argument(
        '--output_file', '-o', required=True,
        help='Path where to output illustrations ids file.'
    )
    args = parser.parse_args()
    extract_illustrations(args.page_file, args.output_file)