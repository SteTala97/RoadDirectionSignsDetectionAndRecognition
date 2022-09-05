"""
Copy-paste the images of the Mapillary Traffic Sign Detection Dataset which are
associated to an annotation file that has the field 'direction-or-information' 
equal to True.	
"""

import glob
import json
import sys
import shutil



# fully annotated data
ANNOTATION_FOLDER = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/fully_annotated/annotations/mtsd_v2_fully_annotated/annotations/'
IMAGES_FOLDERS = [
                  'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/fully_annotated/training_set_0/images/',
				  'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/fully_annotated/training_set_1/images/',
				  'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/fully_annotated/training_set_2/images/',
				  'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/fully_annotated/validation_set/images/',
				  'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/fully_annotated/test_set/images/'
                 ]
DESTINATION_FOLDER = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/direction_or_information_FA/'
## partially annotated data
# ANNOTATION_FOLDER = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/partially_annotated/annotations/mtsd_v2_partially_annotated/annotations/'
# IMAGES_FOLDERS = [
#                   'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/partially_annotated/training_set_0/images/',
# 				    'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/partially_annotated/training_set_1/images/',
# 				    'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/partially_annotated/training_set_2/images/',
# 				    'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/partially_annotated/training_set_3/images/'
# 				   ]
# DESTINATION_FOLDER = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/direction_or_information_PA/'



def main():
	total_files = 0 # total ammount of annotation files in the folder: 'ANNOTATION_FOLDER'
	images_to_take = [] # list of paths of files associated to images that contain 'direction-or-information'
	                    # signs, which will be used to get those images

	# take the annotation files that refer to images that contain 'direction-or-information' signs
	print("\n LOOKING FOR IMAGES WITH PROPERTY 'direction-or-information' EQUAL TO 'True'...\n")
	for file in glob.glob(ANNOTATION_FOLDER + '*.json'):
		
		total_files += 1
		with open(file, 'r') as f:
			annotations = json.load(f)

		# get only the files with flag 'direction-or-information' equal to True	
		for obj_index in range(len(annotations['objects'])):
			if annotations['objects'][obj_index]['properties']['direction-or-information'] == True:
				images_to_take.append(file)
				break


	print(f"\n There is a total of {total_files} annotation files")
	print(f" Of which {len(images_to_take)} refer to an image with direction or indication signs")
	# print(f"\n These are the image names:")
	# if len(images_to_take) > 0:
	# 	for name in images_to_take:
	# 		print(name)


	# Take the images with direction or indication signs and save them to a dedicated folder, along with the annotation file
	print("\n COPYING IMAGES OF DIRECTION OR INFORMATION SIGNS TO THE DEDICATED FOLDER...\n")
	copied = {} # dictionary used to check which images have not been copied
	for folder in IMAGES_FOLDERS:
		for annotation_path in images_to_take:
			dst_path = annotation_path.split('/')[-1]
			dst_path = dst_path.lstrip('annotations\\')
			target_annotation = dst_path
			dst_path = dst_path.rstrip('json')
			original_img = folder + dst_path + 'jpg'
			target_img = DESTINATION_FOLDER + dst_path + 'jpg'
			# save a copy of the image and annotation file in the dedicated 'directrion-or-information' folder
			try:
				shutil.copy(original_img, target_img) # copy image
				copied[dst_path] = dst_path # add the filename to the dictionary to later check if the copy has been successful
				target_annotation = DESTINATION_FOLDER + target_annotation
				shutil.copy(annotation_path, target_annotation) # copy annotation file 
			except:
				pass
	
	# check if all images have been copied
	if len(copied) == len(images_to_take):
		print(f" All of the {len(images_to_take)} images have been successfully copied!")
	else:
		print(f" {len(images_to_take) - len(copied)} of {len(images_to_take)} images have not been copied!")
		# print(" These are the file names:")
		# for name in images_to_take:
		# 	if name not in [filename for filename in copied]:
		# 		print(name)


if __name__ == '__main__':
	main()
