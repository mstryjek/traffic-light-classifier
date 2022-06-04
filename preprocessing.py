import numpy as np
import cv2

IMG_FINAL_SIZE = (75, 40)


def resize_image(image):
  """Resizes an image to a given shape

  Parameters
  ----------
  image -- image to be resized

  Returns
  -------
  resized image
  """
  
  width, height = IMG_FINAL_SIZE[1], IMG_FINAL_SIZE[0]
  
  return cv2.resize(image, (width, height))



def normalize_img(img):
  """Normalizes an img to the (-1, 1) range
  
  Parameters
  ----------
  img -- a numpy array representing an img

  Returns
  -------
  Normalized img  
  """
  return img.astype(np.float64) / 255


def histogram_equalization(img):
  """
  Parameters
  ----------

  img -- a numpy array representing an RGB img

  Returns
  -------

  img of equalized grayscale
  """

  img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  img[:, :, -1] = cv2.equalizeHist(img[:, :, -1])
  
  return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)


def blur(img):
	"""Applies gaussian blur to an img
	Parameters
	----------
	img -- numpy array representing an img
	Returns
	-------
	img with gaussian blur applied
	"""

	# Reducing noise - smoothing
	blurred_img = cv2.medianBlur(img, 3)

	return blurred_img


def preprocess_img(img):
	"""Applies grayscale transformation, equalization and normalization to an img

	Parameters
	----------

	img -- a numpy array representing an img

	Returns
	-------
	preprocessed img

	"""
	# img = histogram_equalization(img)
	# img = blur(img)
	normalized = normalize_img(img)
	return normalized


def pipeline(data):
  resized = np.array(list(map(resize_image, data)))
  preprocessed = np.array(list(map(preprocess_img, resized)))
  #X = np.expand_dims(preprocessed, axis=-1)
  return preprocessed