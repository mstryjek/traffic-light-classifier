import numpy as np
import pandas as pd
import cv2

import tqdm

import argparse
import os
import glob

from typing import Dict



def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the script."""
	parser = argparse.ArgumentParser()

	parser.add_argument('-D', '--directory', type=str, default='CARLA', help='Path to dataset directory')
	parser.add_argument('-W', '--width',     type=int, default=40    , help='Target image width')
	parser.add_argument('-H', '--height',    type=int, default=75    , help='Target image height')

	return parser.parse_args()


def convert_CARLA(data_path: str, width: int, height: int) -> np.ndarray:
	"""
	Convert the CARLA-generated dataset into a numpy array.
	"""
	img_files = glob.glob(os.path.join(data_path, 'train/back/*.png'))

	images_converted = []

	for img_file in tqdm.tqdm(img_files, desc=f'Processing the CARLA dataset...', total=len(img_files)):
		img = cv2.imread(img_file, cv2.IMREAD_COLOR)

		img = cv2.resize(img, (width, height))

		images_converted.append(img)
	
	print('Done processing dataset')

	return np.stack(images_converted, axis=0).astype(np.uint8)


def main() -> None:
	args = parse_args()

	images = convert_CARLA(args.directory, args.width, args.height)

	np.save('data/other.npy', images)


if __name__ == '__main__':
	main()

