#!/usr/bin/env python

from operator import itemgetter
from optparse import OptionParser
import logging
from PIL import Image
from math import ceil, floor
import sys

"""
Copyright 2014 Trailbehind inc

Implements algorithm presented in https://www.crispymtn.com/stories/the-algorithm-for-a-perfectly-balanced-photo-gallery

Linear partition algorithm from 
http://stackoverflow.com/questions/7938809/dynamic-programming-linear-partitioning-please-help-grok/7942946#7942946
"""
def linear_partition(seq, k):
    if k <= 0:
        return []
    n = len(seq) - 1
    if k > n:
        return map(lambda x: [x], seq)

    values = [i[0] for i in seq]
    table, solution = linear_partition_table(values, k)
    k, ans = k-2, []
    while k >= 0:
        ans = [[seq[i] for i in range(solution[n-1][k]+1, n+1)]] + ans
        n, k = solution[n-1][k], k-1
    ans = [[seq[i] for i in range(0, n+1)]] + ans
    return ans


def linear_partition_table(seq, k):
    n = len(seq)
    table = [[0] * k for x in range(n)]
    solution = [[0] * (k-1) for x in range(n-1)]
    for i in range(n):
        table[i][0] = seq[i] + (table[i-1][0] if i else 0)
    for j in range(k):
        table[0][j] = seq[0]
    for i in range(1, n):
        for j in range(1, k):
            table[i][j], solution[i-1][j-1] = min(
                ((max(table[x][j-1], table[i][0]-table[x][0]), x) for x in range(i)),
                key=itemgetter(0))
    return (table, solution)


def create_collage(image_files, result_width, rows):
    """
    Create a collage of images, with <rows> rows, filling all space without cropping
    """

    logging.debug("Creating collage of %d images with %d rows" % (len(image_files), rows))

    if len(image_files) < rows:
        rows = len(image_files)

    images = []
    for filename_or_image in image_files:
        if isinstance(filename_or_image, Image.Image):
            im = filename_or_image
        else:
            im = Image.open(filename_or_image)
        aspect = float(im.size[0])/float(im.size[1])
        images.append((aspect, im))

    partioned_images = linear_partition(images, rows)

    result_height = 0
    for row in partioned_images:
        row_width = float(sum([i[0] for i in row]))
        aspect, image = row[0]
        width_percent = aspect/row_width
        image_width = width_percent * result_width
        scale_ratio = image_width/float(image.size[0])
        result_height += image.size[1] * scale_ratio

    result_size = (result_width, int(result_height))
    logging.debug("Creating result image size: %f*%f" % result_size)
    result = Image.new('RGB', result_size)

    y_position = 0
    for row in partioned_images:
        row_width = float(sum([i[0] for i in row])) #Sum of aspect ratios for all images in row
        x_position = 0
        row_height = 0
        for aspect, image in row:
            width_percent = aspect/row_width
            image_width = width_percent * result_width
            scale_ratio = image_width/float(image.size[0])
            logging.debug(("Image size: %f %f" % image.size) + " percent of row " + str(width_percent) + " new width " + str(image_width) + " scale_ratio " + str(scale_ratio))
            new_size = [int(ceil(i * scale_ratio)) for i in image.size]
            logging.debug("resizing image to " + str(new_size))
            resized_image = image.resize(new_size)
            paste_position = (x_position, y_position)
            logging.debug("pasting image at " + str(paste_position))
            result.paste(resized_image, paste_position)
            x_position += new_size[0]
            row_height = new_size[1]
        y_position += row_height

    return result


def _main():
    usage = "usage: %prog img.jpg img2.jpg ..."
    parser = OptionParser(usage=usage,
     description="Create a collage of images, filling all space, without cropping any images")
    parser.add_option("-d", "--debug", action="store_true", dest="debug",
     help="Turn on debug logging")
    parser.add_option("-o", "--output", action="store", dest="output",
     default=None, help="output to file, if no file is specified image will be displayed")
    parser.add_option("-w", "--width", action="store", dest="width", type="int",
     default=1000, help="Image width")
    parser.add_option("-r", "--rows", action="store", dest="rows", type="int",
     default=None, help="Number of rows, default is <number of images>/2")
    
    (options, args) = parser.parse_args()
 
    logging.basicConfig(level=logging.DEBUG if options.debug else logging.ERROR)
 
    #if len(args) < 2:
    #    logging.error("Error, must specify at least 2 images")
    #    sys.exit(-1)

    if options.rows is not None:
        rows = options.rows
    else:
        rows = len(args)/2

    import os
    import random
    img_dir = "/Users/janivanecky/all_negative"
    image_files = [os.path.join(img_dir, d) for d in os.listdir(img_dir) if d.endswith('.jpeg')]
    random.shuffle(image_files)
    image_files = image_files[:10]
    result = create_collage(image_files, options.width, rows)
    result.show()

    if options.output:
        result.save(options.output)
    else:
        result.show()

if __name__ == "__main__":
    _main()