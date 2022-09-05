import glob
import json
import os


# either on fully annotated data (FA) or partially annotated data (PA)
SRC_FOLDER = 'D:/UNIVERSITA/Magistrale/SecondoAnno/Tesi/Datasets/MapillaryTrafficSignDetection/direction_or_information_FA/'


def main():
	to_keep = [] 
	for file in glob.glob(SRC_FOLDER + '*.jpg'):
		to_keep.append(file)

	n_removed = 0
	for file in glob.glob(SRC_FOLDER + '*.json'):
		if file[:-4]+'jpg' not in to_keep:
			os.remove(file)
			n_removed += 1
			print(f" Files removed: {n_removed}")
			


if __name__ == '__main__':
	main()
