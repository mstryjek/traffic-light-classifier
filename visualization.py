import numpy as np
import cv2


def draw_light_class(img: np.ndarray, cid: int) -> np.ndarray:
	"""Create and publish an image to the debug topic."""
	colors = { ## Colors are in BGR
		0: [0, 255,   0],
		1: [0,   0, 255],
		2: [0, 255, 255],
		3: [255, 0,   0]
	}

	img = img.astype(np.uint8)

	color = tuple(colors[cid])
	## (w, h)
	IMAGE_SHAPE = (200, 400)
	RECT_SHAPE = (45, 120)
	MARGIN = 5
	LIGHT_RADIUS = 8
	GRAY = (80,)*3

	## Resize image and draw rectangles
	img = cv2.resize(img, IMAGE_SHAPE)

	## Gray for contrast
	gray_start = (0, img.shape[0] - (RECT_SHAPE[1] + MARGIN))
	gray_stop = (RECT_SHAPE[0]+MARGIN, img.shape[0])
	img = cv2.rectangle(img, gray_start, gray_stop, GRAY, -1)

	## Black as traffic light background
	black_start = (0, img.shape[0] - (RECT_SHAPE[1]))
	black_stop = (RECT_SHAPE[0], img.shape[0])
	img = cv2.rectangle(img, black_start, black_stop, (0,0,0), -1)

	## Draw lights as circles
	if cid != 3:
		color_circ_idx = cid if cid == 0 else 1 if cid == 2 else 2
		for i in range(3):
			clr = color if i == color_circ_idx else GRAY
			circ_center = (int(RECT_SHAPE[0]/2),
			int(img.shape[0] - RECT_SHAPE[1]/3*i - RECT_SHAPE[1]/6))

			img = cv2.circle(img, circ_center, LIGHT_RADIUS, clr, -1)
	else:
		img = cv2.line(img, (0, img.shape[0]-RECT_SHAPE[1]), (RECT_SHAPE[0], img.shape[0]), (0, 0, 255), 8)

	return img
