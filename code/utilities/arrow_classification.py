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
	# img_th = apply_morfology(img_th)

	# Connected components analysis on thresholded image
	numLabels, labels, stats, centroid = cv.connectedComponentsWithStats(img_th, 4, cv.CV_32S)
	areas = stats[:, cv.CC_STAT_AREA]
	areas[0] = 0
	maxarea = max(areas)
	maxlabel = np.argmax(areas)
	arrow_mask = (labels == maxlabel).astype(np.uint8) * 255
	# Check if the CC intersects the "central patch" of the image, otherwise take the max CC from the inverse of the binary mask;
	# the "central patch" is a centered patch that spans n_rows/10 and n_cols/10 pixels in both directions
	# rows_patch = r // 10 if (r // 20) > 0 else 1
	# cols_patch = c // 10 if (c // 20) > 0 else 1
	# if np.sum(arrow_mask[r//2-rows_patch:r//2+rows_patch, c//2-cols_patch:c//2+cols_patch]) == 0:
	if arrow_mask[r//2, c//2] == 0: # check only for the central pixel
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

	# Rotate the arrow mask according to the two vectors to allign the arrow vertically
	angle_eig1 = -theta + 90 # angle that alligns the first eigenvector
	mask_eig1, diff_eig1 = check_symmetry(arrow_mask, angle_eig1)
	angle_eig2 = -theta     # angle that alligns the second eigenvector
	mask_eig2, diff_eig2 = check_symmetry(arrow_mask, angle_eig2)

	# Consider also that the shape of the arrow may "trick" the PCA and give a 45° angle shift
	angle_eig1_45pos = angle_eig1 - 45 # rotate the first eigenvector 45° less (this is the vector at +45° from the eigenvector1)
	mask_eig1_45pos, diff_eig1_45pos = check_symmetry(arrow_mask, angle_eig1_45pos)
	angle_eig1_45neg = angle_eig1 + 45 # rotate the first eigenvector 45° more (this is the vector at -45° from the eigenvector1)
	mask_eig1_45neg, diff_eig1_45neg = check_symmetry(arrow_mask, angle_eig1_45neg)

	# Get the most symmetric image rotation
	if diff_eig1 < diff_eig2 and diff_eig1 < diff_eig1_45pos and diff_eig1 < diff_eig1_45neg:               # mask_eig1
		orientation        = theta
		total_rotation     = angle_eig1
		most_symmetryc_img = mask_eig1
		color_eig1         = (0, 255, 0)
		color_eig2 = color_eig45 = color_eig135 = (0, 0, 255)
	elif diff_eig2 < diff_eig1 and diff_eig2 < diff_eig1_45pos and diff_eig2 < diff_eig1_45neg:             # mask_eig2
		orientation        = theta + 90
		total_rotation     = angle_eig2
		most_symmetryc_img = mask_eig2
		color_eig2         = (0, 255, 0)
		color_eig1 = color_eig45 = color_eig135 = (0, 0, 255)
	elif diff_eig1_45pos < diff_eig2 and diff_eig1_45pos < diff_eig1 and diff_eig1_45pos < diff_eig1_45neg: # mask_eig1_45pos
		orientation        = theta + 45
		total_rotation     = angle_eig1_45pos
		most_symmetryc_img = mask_eig1_45pos
		color_eig45        = (0, 255, 0)
		color_eig2 = color_eig1 = color_eig135 = (0, 0, 255)
	else:																				                    # mask_eig1_45neg
		orientation        = theta - 45
		total_rotation     = angle_eig1_45neg
		most_symmetryc_img = mask_eig1_45neg
		color_eig135       = (0, 255, 0)
		color_eig2 = color_eig45 = color_eig1 = (0, 0, 255)

	# Histogram of sum of rows
	hist = []
	height = most_symmetryc_img.shape[0]
	width = most_symmetryc_img.shape[1]
	half_width = width // 2 - 1
	for row in range(height):
		# While iterating through each row, "fill the empty space" between two specular white pixels
		l_flag = r_flag = False
		r_px = l_px = 0
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
	max_ = max(hist)
	nhist = [float(x) / max_ for x in hist] # normalized histogram
	w = 7 if len(nhist) > 100 else 3
	nhist = np.convolve(nhist, np.ones(w), 'valid') / w # smoothed histogram

	# Get the portion of plot with the longest ascending slope (the bottom part is reversed to be compared with the top one)
	slopes_top = {}
	slopes_bottom = {}
	key_top = 0
	key_bottom = 0
	slopes_top[key_top] = 0
	slopes_bottom[key_bottom] = 0
	len_nhist = len(nhist)
	len_nhist_halved = len_nhist // 2
	for i in range(1, len_nhist_halved):
		if nhist[i] > nhist[i-1]:
			slopes_top[key_top] += 1
		else:
			key_top += 1
			slopes_top[key_top] = 0
		if nhist[len_nhist - i - 1] > nhist[len_nhist - i]:
			slopes_bottom[key_bottom] += 1
		else:
			key_bottom += 1
			slopes_bottom[key_bottom] = 0

	if max(slopes_top.values()) > max(slopes_bottom.values()):
		mask_points_up = True
	else:
		mask_points_up = False

	# Infer orientation
	rot_tollerance = 20 # this is approximately half of the amplitude of an orientation interval (which is 45°)
	if abs(orientation) > 70 and abs(orientation) < 110:
		if mask_points_up:
			direction_res = "↓" if abs(total_rotation) > (90 + rot_tollerance) else "↑"
		else:
			direction_res = "↑" if abs(total_rotation) > (90 + rot_tollerance) else "↓"
	elif abs(orientation) > 155 or abs(orientation) < 25:
		if mask_points_up:
			direction_res = "←" if abs(total_rotation) > (90 + rot_tollerance) else "→"
		else:
			direction_res = "→" if abs(total_rotation) > (90 + rot_tollerance) else "←"
	elif orientation >= 25 and orientation <= 65 or orientation <= -115 and orientation >= -155 or orientation >= 205 and orientation <= 245:
		if mask_points_up:
			direction_res = "↙" if abs(total_rotation) > (90 + rot_tollerance) else "↗"
		else:
			direction_res = "↗" if abs(total_rotation) > (90 + rot_tollerance) else "↙"
	else:
		if mask_points_up:
			direction_res = "↘" if abs(total_rotation) > (90 + rot_tollerance) else "↖"
		else:
			direction_res = "↖" if abs(total_rotation) > (90 + rot_tollerance) else "↘"

	return direction_res
