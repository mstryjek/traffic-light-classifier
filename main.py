import keras
import cv2
import numpy as np
import matplotlib.pyplot as plt

import os
import glob

from preprocessing import pipeline
from visualization import draw_light_class


def main() -> None:
	IMG_PATH = 'test_resources'
	MODEL_PATH = 'models'
	
	model = keras.models.load_model(os.path.join(MODEL_PATH, 'TLC.h5'))
	image_files = glob.glob(os.path.join(IMG_PATH, '*.jpg'))
	images = []

	for img in image_files:	
		images.append(cv2.imread(img, cv2.IMREAD_COLOR))
	
	images_preprocessed = pipeline(images)

	predictions = model.predict(images_preprocessed)
	classes = np.argmax(predictions, axis=-1)

	drawn_images = []
	for img, cls_ in zip(images, classes):
		img_drawn = draw_light_class(img, cls_)
		drawn_images.append(img_drawn)
	
	final_img = np.concatenate(drawn_images, axis=1)

	plt.imshow(cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB))
	plt.show()


if __name__ == '__main__':
	main()

