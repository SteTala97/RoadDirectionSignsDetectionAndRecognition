# This script is used to create a shell file which aims to create dynamic links
# for the image folders on the University of Bicocca's GPU server.
# NOTE: the .sh files are later manually converted from .txt

import glob


FOLDERS = ['train']#, 'val', 'test']
COMMON_PATH = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/code/yolov7/data/MTSD-fully_annotated_025'
SHELL_FILENAMES = ['ln_train.txt']#, 'ln_val.txt', 'ln_test.txt']
WINDOWS_LINE_ENDING = b'\r\n'
UNIX_LINE_ENDING = b'\n'


def main():
	for i in range(1):
		path = COMMON_PATH + '/' + FOLDERS[i] + '/images/*.jpg'
		img_files = [file for file in glob.glob(path)]
		for img_file in img_files:
			with open(SHELL_FILENAMES[i], 'a') as f:
				# the names of the images are taken from index 1 and not 0 to remove the initial '/' character
				f.write(f"ln -s /datasets/mapillary/mtsd_v2_fully_annotated/images/{img_file.lstrip(path)[1:]} -- {img_file.lstrip(path)[1:]}\n")

	# then convert the Windows-like line endings to make them UNIX-like line endings 
	for i in range(1):
		with open(SHELL_FILENAMES[i], 'rb') as f:
			original_content = f.read()
		# Windows →	UNIX
		modified_content = original_content.replace(WINDOWS_LINE_ENDING, UNIX_LINE_ENDING)
		# write back to file
		with open(SHELL_FILENAMES[i], 'wb') as f:
			f.write(modified_content)

if __name__ == '__main__':
	main()
