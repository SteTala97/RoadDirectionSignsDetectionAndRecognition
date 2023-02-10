import cv2 as cv
import numpy as np


def rotate(img, angle):
	# Rotate input image adapting the dimensions to the content to not cut out parts of it
	(r, c) = img.shape
	(cX, cY) = (c // 2, r // 2)
	M = cv.getRotationMatrix2D((cX, cY), angle, 1)
	cos = np.abs(M[0, 0])
	sin = np.abs(M[0, 1])
	newW = int((r * sin) + (c * cos))
	newH = int((r * cos) + (c * sin))
	M[0, 2] += (newW / 2) - cX
	M[1, 2] += (newH / 2) - cY

	return cv.warpAffine(img, M, (newW, newH))


def crop(img):
	# Crop image to fit the arrow ROI
	r, c = img.shape
	coordinates_grid = np.ones((2, r, c), dtype=np.int16)
	coordinates_grid[0] = coordinates_grid[0] * np.array([range(r)]).T
	coordinates_grid[1] = coordinates_grid[1] * np.array([range(c)])
	mask = img[:, :] != 0
	non_zeros = np.hstack((coordinates_grid[0][mask].reshape(-1, 1),
						   coordinates_grid[1][mask].reshape(-1, 1)))
	left =   min([non_zeros[i, 0] for i in range(len(non_zeros))]) + 1
	right =  max([non_zeros[i, 0] for i in range(len(non_zeros))]) + 1
	top =    min([non_zeros[i, 1] for i in range(len(non_zeros))]) + 1
	bottom = max([non_zeros[i, 1] for i in range(len(non_zeros))]) + 1
	img = img[left:right, top:bottom]

	return img


def apply_morfology(img):
	kernel = np.ones((9, 9), dtype=np.uint8)
	eroded = cv.erode(img, kernel, iterations=1)
	kernel = np.ones((5, 5), dtype=np.uint8)
	return cv.dilate(eroded, kernel, iterations=1)


def check_symmetry(img, angle):
	# Check vertical symmetry
	img = rotate(img, angle) if angle != 0 else img
	# img = cv.threshold(img, 10, 255, cv.THRESH_BINARY)[1]
	img = crop(img)
	new_shape = img.shape
	odd_val = 1 if ((new_shape[1] % 2) != 0) else 0 # this ensures that in case the dimension is odd, left and right halves are the same size
	l_half = img[:, :new_shape[1] // 2]
	r_half = np.flip(img[:, new_shape[1] // 2 + odd_val:], axis=1)
	diff = cv.absdiff(l_half, r_half)
	diff_val = np.sum(diff)
	if diff_val is None:
		diff_val = new_shape[0] * new_shape[1]

	return img, diff_val


def getOrientation(pts, img_, dst):
	sz = len(pts)
	data_pts = np.empty((sz, 2), dtype=np.float64)
	for i in range(data_pts.shape[0]):
		data_pts[i,0] = pts[i,0,0]
		data_pts[i,1] = pts[i,0,1]
	# Perform PCA analysis
	mean = np.empty((0))
	mean, eigenvectors, eigenvalues = cv.PCACompute2(data_pts, mean)
	# Store the center of the object
	cntr = (int(mean[0,0]), int(mean[0,1]))
	# Draw results
	p1 = (cntr[0] + 0.02 * eigenvectors[0,0] * eigenvalues[0,0], cntr[1] + 0.02 * eigenvectors[0,1] * eigenvalues[0,0])
	p2 = (cntr[0] - 0.02 * eigenvectors[1,0] * eigenvalues[1,0], cntr[1] - 0.02 * eigenvectors[1,1] * eigenvalues[1,0])
	# Get the orientation
	angle = np.arctan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
	
	return angle, dst, cntr


def classify_arrow(img):
	r, c = np.shape(img)[:-1]

	# Thresholding
	img_blur = cv.GaussianBlur(img, (5, 5), 0)
	img_gray = cv.cvtColor(img_blur, cv.COLOR_BGR2GRAY)
	img_th = cv.threshold(img_gray, 0, 255, cv.THRESH_OTSU)[1]

	# Connected components analysis on thresholded image
	numLabels, labels, stats, centroid = cv.connectedComponentsWithStats(img_th, 4, cv.CV_32S)
	areas = stats[:, cv.CC_STAT_AREA]
	areas[0] = 0
	maxarea = max(areas)
	maxlabel = np.argmax(areas)
	arrow_mask = (labels == maxlabel).astype(np.uint8) * 255
	if not(arrow_mask[r//2, c//2]):
		# Connected components analysis on inverted thresholded image 
		img_th_inv = 255 - img_th
		numLabels, labels, stats, centroid_inv = cv.connectedComponentsWithStats(img_th_inv, 4, cv.CV_32S)
		areas = stats[:, cv.CC_STAT_AREA]
		areas[0] = 0
		maxarea = max(areas)
		maxlabel_inv = np.argmax(areas)
		arrow_mask = (labels == maxlabel_inv).astype(np.uint8) * 255

	# PCA:
	contours, _ = cv.findContours(arrow_mask, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
	angle, res, cntr = getOrientation(contours[0], arrow_mask, img)
	theta = int(-(angle / np.pi * 180)) # rad to deg (and invert the sign)

	# Bring the angle in the range [0, 360] if not already so
	theta = 360 + theta if theta < 0 else theta

	# Set the increment to keep each of the four directions in the range [0, 360]
	increment = 45 if theta < 135 else -45

	# Rotate the arrow mask according to the two vectors to allign the arrow vertically
	og_angle_eig1 = theta
	angle_eig1 = 90 - theta # angle that alligns the first eigenvector
	mask_eig1, diff_eig1 = check_symmetry(arrow_mask, angle_eig1)
	og_angle_eig2 = theta + increment * 2
	angle_eig2 = 90 - og_angle_eig2 # angle that alligns the second eigenvector
	mask_eig2, diff_eig2 = check_symmetry(arrow_mask, angle_eig2)

	# Consider also that the shape of the arrow may "trick" the PCA and give a 45° angle shift
	og_angle_45_1 = theta + increment
	angle_45_1 = 90 - og_angle_45_1 # rotate the first eigenvector 45° less (this is the vector at +45° from the eigenvector1)
	mask_45_1, diff_eig1_45pos = check_symmetry(arrow_mask, angle_45_1)
	og_angle_45_2 = theta + increment * 3
	angle_45_2 = 90 - og_angle_45_2 # rotate the first eigenvector 45° more (this is the vector at -45° from the eigenvector1)
	mask_45_2, diff_eig1_45neg = check_symmetry(arrow_mask, angle_45_2)

	# Get the most symmetric image rotation
	if diff_eig1 < diff_eig2 and diff_eig1 < diff_eig1_45pos and diff_eig1 < diff_eig1_45neg:               # mask_eig1
		orientation        = og_angle_eig1 # original orientation
		total_rotation     = angle_eig1 # rotation applied to allign the mask vertically
		most_symmetryc_img = mask_eig1
	elif diff_eig2 < diff_eig1 and diff_eig2 < diff_eig1_45pos and diff_eig2 < diff_eig1_45neg:             # mask_eig2
		orientation        = og_angle_eig2 # original orientation
		total_rotation     = angle_eig2 # rotation applied to allign the mask vertically
		most_symmetryc_img = mask_eig2
	elif diff_eig1_45pos < diff_eig2 and diff_eig1_45pos < diff_eig1 and diff_eig1_45pos < diff_eig1_45neg: # mask_45_1
		orientation        = og_angle_45_1 # original orientation
		total_rotation     = angle_45_1 # rotation applied to allign the mask vertically
		most_symmetryc_img = mask_45_1
	else:																				                    # mask_45_2
		orientation        = og_angle_45_2 # original orientation
		total_rotation     = angle_45_2 # rotation applied to allign the mask vertically
		most_symmetryc_img = mask_45_2

	# Histogram of sum of rows
	hist = []
	height = most_symmetryc_img.shape[0]
	width = most_symmetryc_img.shape[1]
	half_width = width // 2 - 1
	skip_counter = 0 # counter used to "skip" a certain number of rows (see below in the nested for loop)
	for row in range(height):
		# While iterating through each line, "fill the empty space" between two specular white pixels
		l_flag = r_flag = False
		r_px = l_px = 0
		skip_counter += 1
		if skip_counter == 3: # perform the following only each 3 rows
			skip_counter = 0
			for col in range(half_width):
				if not(l_flag) and most_symmetryc_img[row, col] > 0:
					l_flag = True
					l_px = col
				if not(r_flag) and most_symmetryc_img[row, width-col-1] > 0:
					r_flag = True
					r_px = width-col-1
				if l_flag and r_flag:
					most_symmetryc_img[row, l_px:r_px] = 255
					break
		# Now sum the current row
		hist.append(np.sum(most_symmetryc_img[row, :]))

	w = 9 if len(hist) > 100 else 3
	hist = np.convolve(hist, np.ones(w), 'valid') / w # smoothed histogram

	# Get the portion of plot with the longest ascending slope (the bottom part is reversed to be compared with the top one)
	slopes_top = {}
	slopes_bottom = {}
	key_top = 0
	key_bottom = 0
	slopes_top[key_top] = 0
	slopes_bottom[key_bottom] = 0
	len_hist = len(hist)
	len_hist_halved = len_hist // 2
	for i in range(1, len_hist_halved):
		if hist[i] > hist[i-1]:
			slopes_top[key_top] += 1
		else:
			key_top += 1
			slopes_top[key_top] = 0
		if hist[len_hist - i - 1] > hist[len_hist - i]:
			slopes_bottom[key_bottom] += 1
		else:
			key_bottom += 1
			slopes_bottom[key_bottom] = 0

	if max(slopes_top.values()) > max(slopes_bottom.values()):
		mask_direction = "up"
	else:
		mask_direction = "down"

	# Determine the angle of the orientation; if needed, invert the angle with respect to the goniometric circumference
	if mask_direction == "up":
		vertex_orientation = orientation
	else:
		vertex_orientation = orientation+180 if orientation < 180 else orientation-180

	# Discretize the result into the 8 directions considered
	if vertex_orientation > 338 or vertex_orientation < 22: # "right" direction
		direction_res = "right"
	elif vertex_orientation >= 22 and vertex_orientation <= 68: # "north-east" direction
		direction_res = "north-east"
	elif vertex_orientation > 68 and vertex_orientation < 112: # "up" direction
		direction_res = "up"
	elif vertex_orientation >= 112 and vertex_orientation <= 158: # "north-west" direction
		direction_res = "north-west"
	elif vertex_orientation > 158 and vertex_orientation < 202: # "left" direction
		direction_res = "left"
	elif vertex_orientation >= 202 and vertex_orientation <= 248: # "south-west" direction
		direction_res = "south-west"
	elif vertex_orientation > 248 and vertex_orientation < 292: # "down" direction
		direction_res = "down"
	elif vertex_orientation >= 292 and vertex_orientation <= 338: # "south-east" direction
		direction_res = "south-east"

	return direction_res
