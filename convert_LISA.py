import numpy as np
import pandas as pd
import cv2

import tqdm

import argparse
import os

from typing import Dict


def parse_args() -> argparse.Namespace:
	"""Parse command-line arguments for the script."""
	parser = argparse.ArgumentParser()

	parser.add_argument('-D', '--directory', type=str, default='LISA', help='Path to dataset directory')
	parser.add_argument('-W', '--width',     type=int, default=40    , help='Target image width')
	parser.add_argument('-H', '--height',    type=int, default=75    , help='Target image height')

	return parser.parse_args()


def convert_LISA(data_path: str, width: int, height: int, clip_type: str, num_clips: int) -> Dict[str, np.ndarray]:
	"""
	Convert the LISA dataset from its original format to an .npy file.
	Args:
	---
	- `data_path`: Path to dataset, absolute or relative
	- `width`: Target image width
	- `height`: Target image height
	- `clip_type`: Either 'day' or 'night'
	- `num_clips`: Number of 'day' or 'night' clips (13 for 'day', 5 for 'night')
	"""
	## Helper prefixes
	annotation_prefix = os.path.join(data_path, f'Annotations/Annotations/{clip_type}Train/{clip_type}Clip')
	data_prefix = os.path.join(data_path, f'{clip_type}Train/{clip_type}Train/{clip_type}Clip')

	## Helper remaps because of weird LISA names
	tag_remaps = {
		'go': 'green',
		'stop': 'red',
		'warning': 'yellow'
	}

	light_classes = [v for v in tag_remaps.values()]

	image_arrays_out = {cls_: [] for cls_ in light_classes}

	for subdir_idx in tqdm.tqdm(range(1, num_clips+1), desc=f'Processing {clip_type} clips...', total=num_clips):
		## Traffic light images for this clip (subdirectory)
		clip_class_images = {
			'green': [],
			'red': [],
			'yellow': []
		}

		## Read in annotations from seperate .csv file
		anno_file = os.path.join(annotation_prefix + str(subdir_idx), 'frameAnnotationsBOX.csv')
		annotations = pd.read_csv(anno_file, sep=';')

		## Iterate through every image for this class
		for _, row in tqdm.tqdm(annotations.iterrows(), desc=f'Processing {clip_type}Clip{subdir_idx}...', total=annotations.shape[0]):
			img = cv2.imread(os.path.join(os.path.join(data_prefix + str(subdir_idx), 'frames/'), row['Filename'].split('/')[-1]), cv2.IMREAD_COLOR)
			
			tag = row['Annotation tag']

			if img is None or tag not in tag_remaps.keys():
				continue

			## Bounding box coordinates
			coords = [[int(row['Upper left corner Y']),
					   int(row['Upper left corner X'])],
					  [int(row['Lower right corner Y']),
					   int(row['Lower right corner X'])]]

			## Pull traffic light only from camera frame
			traffic_light_img = img[coords[0][0]:coords[1][0], coords[0][1]:coords[1][1], :]

			## Resize to target shape
			traffic_light_img = cv2.resize(traffic_light_img, (width, height))

			## Pull image class and add it to the appropriate list
			img_class = tag_remaps[row['Annotation tag']]
			clip_class_images[img_class].append(traffic_light_img)

		for cls_ in light_classes:
			if len(clip_class_images[cls_]):
				image_arrays_out[cls_].append(np.stack(clip_class_images[cls_], axis=0))

	for cls_ in light_classes:
		image_arrays_out[cls_] = np.vstack(image_arrays_out[cls_])

	return image_arrays_out

def main() -> None:
	args = parse_args()
	day = convert_LISA(args.directory, args.width, args.height, 'day',  13)
	night = convert_LISA(args.directory, args.width, args.height, 'night', 5)

	green = np.vstack([day['green'], night['green']])
	red = np.vstack([day['red'], night['red']])
	yellow = np.vstack([day['yellow'], night['yellow']])

	np.save('data/green.npy', green)
	np.save('data/red.npy', red)
	np.save('data/yellow.npy', yellow)


if __name__ == '__main__':
	main()
