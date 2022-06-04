import cv2
import numpy as np

def _force_RGB_array(img):
    """
    Ensures image is in the form of a (h, w, 3) array.
    Not intended for standalone use.

    Args
    ----------
    img - an RGB image or corresponding array, a grayscale image or corresponding array, or
    an image-like object convertible into numpy.array.

    Returns
    --------
    An array with shape (h, w, 3), where h is the height and w is the width of the original array
    """
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    if len(img.shape) == 3:
        return img
    elif len(img.shape) == 2:
        img = np.dstack([img, img, img])
        return img


def fliph(img):
    """
    Flips the image horizontally (against the central vertical axis)

    Args
    ----------
    img - image to flip

    Returns
    -------
    Array corresponding to the input image, flipped horizontally.
    """
    img = _force_RGB_array(img)
    img = np.fliplr(img)
    return img


def zoom(img, fac=.9):
    """
    Zooms into the center of an image by fac.

    Args
    ----------
    img - image to zoom
    fac - percantage of the image to zoom; 2-long list or tuple of floats, or float. If float, image is zoomed by fac in each direction.
    If list or tuple, zoomed by fac[0]*height vertically and fac[1]*width horizontally. In both cases floats
    should be between 0 and 1.
    
    Returns
    -------
    Array corresponding to the central (height*fac, width*fac) region of the image.
    """
    img = _force_RGB_array(img)
    if not isinstance(fac, tuple) and not isinstance(fac, list):
        fac = (fac, fac)
    if fac[0] > 1:
        fac = (1, fac[1])
    if fac[1] > 1:
        fac = (fac[0], 1)
    if fac[0] <= 0:
        fac = (.5, fac[1])
    if fac[1] <= 0:
        fac = (fac[0], .5)
    h, w, _ = img.shape
    bounds_h = [int(h/2-h*(fac[0]/2.)), int(h/2+h*(fac[0]/2.))]
    bounds_w = [int(w/2-w*(fac[1]/2.)), int(w/2+w*(fac[1]/2.))]
    img = img[bounds_h[0]:bounds_h[1], bounds_w[0]:bounds_w[1], :]
    return img
    

def noisy(img, amt):
    """
    Adds noise to the image.
    
    Args
    ----------
    img - image to augment
    amt - amount of noise to use. Float between 0 and 1, 1 being maximum image brightness.
    
    Returns
    -------
    Input image in the form of a 3-dimensional array, with added noise.
    """

    img = _force_RGB_array(img)
    noise = np.random.uniform(size=img.shape) * 255
    img = img + noise*amt
    return img


def mirrorleft(img):
    """
    Mirrors the left half of the image to the right.
    
    Args 
    ---------
    img - image to mirror

    Returns
    ------
    Array corresponding to image, symmetrical against the vertical axis.
    """
    img = _force_RGB_array(img)
    flipclone = np.fliplr(img)
    bnd = img.shape[1] // 2
    img[:, bnd:, :] = flipclone[:, bnd:, :]
    return img


def mirrorright(img):
    """
    Mirrors the right half of the image to the left.

    Args 
    ---------
    img - image to mirror

    Returns
    ------
    Array corresponding to image, symmetrical against the vertical axis.
    """
    img = _force_RGB_array(img)
    flipclone = np.fliplr(img)
    bnd = img.shape[1] // 2
    img[:, :bnd, :] = flipclone[:, :bnd, :]
    return img



def shiftdown(img, px=4):
    """
    Shifts the image down by px, with the gap at the top filled by stretching the topmost row of pixels.

    Parameteres
    ----------
    img - image to shift
    px - number of pixels to shift by

    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = _force_RGB_array(img)
    img = np.roll(img, px, axis=0)
    if px > 0:
        for i in range(px):
            img[i, :, :] = img[px, :, :]
    return img


def shiftup(img, px=4):
    """
    Shifts the image up by px, with the gap at the bottom filled by stretching the bottom row of pixels.
    Parameteres
    ----------
    img - image to shift
    px - number of pixels to shift by
    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = _force_RGB_array(img)
    img = np.roll(img, -px, axis=0)
    if px > 0:
        for i in range(px):
            img[img.shape[0] - i - 1, :, :] = img[img.shape[0] - px, :, :]
    return img


def shiftleft(img, px=3):
    """
    Shifts the image left by px, with the gap at the right filled by stretching the rightmost row of pixels.

    Parameteres
    ----------
    img - image to shift
    px - number of pixels to shift by

    Returns
    -------
    Array corresponding to the shifted image.
    """

    px = abs(px)
    img = _force_RGB_array(img)
    img = np.roll(img, -px, axis=1)
    if px > 0:
        for i in range(px):
            img[:, img.shape[1] - i - 1, :] = img[:, img.shape[1] - px, :]
    return img


def shiftright(img, px=3):
    """
    Shifts the image right by px, with the gap at the left filled by stretching the leftmost row of pixels.
    Parameteres
    ----------
    img - image to shift
    px - number of pixels to shift by
    Returns
    -------
    Array corresponding to the shifted image.
    """
    px = abs(px)
    img = _force_RGB_array(img)
    img = np.roll(img, px, axis=1)
    if px > 0:
        for i in range(px):
            img[:, i, :] = img[:, px, :]
    return img



def random_augment(img):
  """ 
  Choose a random type of image augmentation

  Paramteres
  ----------

  img - image to augment

  Returns
  -------

  Randomly augmented image

  """
  augs = [shiftdown, shiftup, shiftright, shiftleft, fliph, zoom]#, mirrorleft, mirrorright]
  func = choice(augs)
  if func in [shiftup, shiftdown, shiftright, shiftleft]:
    px = randint(1, 5)
    return func(img, px)
  elif func == noisy:
      amt = np.random.uniform(0., .05)
      return func(img, amt)
  elif func == zoom:
    fac = (random()-0.5)/10. + 1. ## [0.95, 1.05)
    return func(img, fac)
  else:
    return func(img)
    



def augment_data(data, labels, num_samples_dict): ## Deleted max_samples param
  """
  Apply data_augment function to the dataset, create new samples.
  For each class, create new augmented samples and add them until num_samples == 0.
  Concatenate original and augmented data.
  """

  new_data_array = []
  new_labels_array = []

  for c, num_samples in num_samples_dict.items():
    print(f'\nGenerating samples for class {c}')
    # Setting only data with current labels
    current_class_images = data[labels == c]
    while num_samples > 0:
      if num_samples % 100 == 0:
        print(c, num_samples)
      # Choosing a random image with a specific class
      random_image = choice(current_class_images)
      
      aug_image = np.array(random_augment(random_image), dtype=np.uint8) ## Changed data_augment to random_augment
      
      new_data_array.append(cv2.resize(aug_image, (IMG_FINAL_SIZE[1], IMG_FINAL_SIZE[0])))
      new_labels_array.append(c)
      num_samples -= 1
    
  print('Shape OK' if all([img.shape == IMG_FINAL_SIZE for img in new_data_array]) else 'Some samples have incorrect shape!')

  np_new_data_array = np.stack(new_data_array, axis=0)
  np_new_labels_array = np.array(new_labels_array)  


  print(f'Created {np_new_data_array.shape[0]} training samples with {np_new_labels_array.shape[0]} labels')

  data_aug = np.concatenate((data, np_new_data_array))       
  labels_aug = np.concatenate((labels, np_new_labels_array))

  print(f'After concatenation with the original dataset: {data_aug.shape[0]} training samples with {labels_aug.shape[0]} labels')
 
  return data_aug, labels_aug
