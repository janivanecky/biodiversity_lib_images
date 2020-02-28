import argparse
from multiprocessing.pool import ThreadPool
import os
import subprocess
import threading
import tqdm

BIODIVERSITY_LIB_URL = 'https://www.biodiversitylibrary.org/pageimage'

def mkdir_p(dir):
    try:
        os.mkdir(dir)
    except OSError as e:
        pass

def download_illustrations(
    illustrations_file, output_dir, start_index=0, max_count=None, thread_count=64
    ):
    # Create output directory.
    mkdir_p(output_dir)

    # Load ilustration ids.
    with open(illustrations_file) as f:
        illustrations = f.readlines()
        illustrations = [i.rstrip('\n') for i in illustrations]

    # Set up start and end indices.
    assert start_index < len(illustrations), "start_index has to be lower than number of illustrations."
    end_index = -1
    if max_count is not None:
        end_index = start_index + max_count
        end_index = min(end_index, len(illustrations))

    # Fetch sub-list of illustrations to download.
    illustrations = illustrations[start_index:end_index]

    # Set up counter.
    lock = threading.Lock()
    pbar = tqdm.tqdm(total=len(illustrations))

    def download_img(img_id):
        # Get output image path and image url.
        output_img_name = '{}/{}.jpeg'.format(output_dir, img_id)
        img_url = '{}/{}'.format(BIODIVERSITY_LIB_URL, img_id)

        # Only download if the image doesn't exist yet.
        if not os.path.exists(output_img_name):
            try:
                subprocess.check_call(['curl', img_url, '-o', output_img_name, '--silent'])
            except:
                print("Failed to fetch illustrations with id {}".format(img_id))

        # Update the counter.
        lock.acquire()
        pbar.update(1)
        lock.release()

    # Download illustrations in parallel.
    pool = ThreadPool(thread_count)
    pool.map(download_img, illustrations)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download illustration files.')
    parser.add_argument(
        'illustrations_file',
        help='Path to illustrations file, as output by extract_illustrations.py'
    )
    parser.add_argument(
        '--output_dir', '-o', required=True,
        help='Path to directory where illustrations will be stored.'
    )
    parser.add_argument(
        '--start_index', '-s', default=0, type=int,
        help='Index of first illustration to download.'
    )
    parser.add_argument(
        '--max_count', '-m', type=int,
        help='Max number of illustrations to download.'
    )
    parser.add_argument(
        '--thread_count', '-t', default=64, type=int,
        help='Number of threads to use to download illustrations in parallel.'
    )
    args = parser.parse_args()
    download_illustrations(
        args.illustrations_file, args.output_dir, args.start_index,
        args.max_count, args.thread_count
    )