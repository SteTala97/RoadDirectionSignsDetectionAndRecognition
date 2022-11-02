"""
	Convert annotation files from the original format of the Mapillary TSD 
	Dataset, which are stored as json files in which the bounding boxes are
	represented in a PascalVOC-like manner, to YOLO format.
	NOTE: both images and annotation files must be in the same folder
"""

import cv2 as cv
import json
import glob


ANNOTATION_FOLDER = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/direction_or_information/'
SHOW_ANNOTATIONS = True
SAVE_ANNOTATION = False
GET_LABELS = False


if __name__ == "__main__":
	src_path = ANNOTATION_FOLDER + '*.json'
	# do the following for each annotation file in the folder 'ANNOTATION_FOLDER'
	for annotation in glob.glob(src_path):
		# read json file
		with open(annotation, 'r') as file:
			data = json.load(file)

		# get all the different road sign categories, specified in the "label" field
		if GET_LABELS:
			labels = {}
			for obj in data['objects']:
				labels[obj['label']] = obj['label']

		# get image resolution to normalize the bbox coordinates
		cols = int(data['width'])
		rows = int(data['height'])

		if SHOW_ANNOTATIONS:
			img = cv.imread(annotation[:-4] + 'jpg')

		# get the coordinates for each bounding box
		yolo_string = ""
		
		for obj in data['objects']:
			# get bbox coordinates from file
			x1 = int(obj['bbox']['xmin'])
			y1 = int(obj['bbox']['ymin'])
			x2 = int(obj['bbox']['xmax'])
			y2 = int(obj['bbox']['ymax'])
			# perform the conversion to YOLO annotation format
			p_min = (x1, y1)
			p_max = (x2, y2)
			# center of the bounding box
			x_center = (p_min[0] + p_max[0]) / 2
			y_center = (p_min[1] + p_max[1]) / 2
			# width and height of the bounding box
			width =  p_max[0] - p_min[0]
			height = p_max[1] - p_min[1]
			# normalize in [0, 1] with respect to the image resolution
			x_center /= cols
			y_center /= rows
			width  /= cols
			height /= rows

			# compose the final annotation string for the current object;
			# the class indexes are: 
			# '0' (direction_or_information) : the traffic sign is a direction or information sign
			# '1' (other) : the traffic sign is any other kind of traffic sign
			class_idx = 0
			if obj['properties']['direction-or-information'] == False:
				class_idx = 1
			yolo_string += f'{class_idx} ' + str(x_center) + ' ' + str(y_center) + ' ' + \
								             str(width)    + ' ' + str(height)   + '\n'

			# draw a bounding box around each object 
			if SHOW_ANNOTATIONS:
				# re-convert the coordinates just to make sure everything's ok
				x_min = int((x_center * cols) - (width  * cols) / 2)
				y_min = int((y_center * rows) - (height * rows) / 2)
				x_max = int((x_center * cols) + (width  * cols) / 2)
				y_max = int((y_center * rows) + (height * rows) / 2)
				cv.rectangle(img, (x_min, y_min), (x_max, y_max), (50, 50, 255), 5)

		# save the new txt annotation file to memory
		if SAVE_ANNOTATION:
			with open(annotation[:-4] + 'txt', 'w+') as new_annotation:
				new_annotation.write(yolo_string)

		# show the original image with the bounding boxes drawn around each object
		if SHOW_ANNOTATIONS:
			cv.namedWindow("direction-or-information", cv.WINDOW_GUI_EXPANDED)
			cv.imshow("direction-or-information", img)
			if 0xFF & cv.waitKey() == 27:
				break
			cv.destroyAllWindows()

	# print all the different traffic sign labels present in the files considered
	if GET_LABELS:
		print("")
		for label in labels:
			print(label)
