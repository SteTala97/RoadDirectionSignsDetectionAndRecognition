import matplotlib.pyplot as plt
import numpy as np


EPOCHS = 300
# location of the files 'results.txt'
RESULTS_FILES = ['1.txt', '05.txt', '025.txt']
# these are the indices corresponding to the order of the statistichs in the 'results.txt' file
P = 8             # Precision
R = 9             # Recall
MAP50 = 10        # mAP@.5
MAP = 11          # mAP@0.5:.95


def main():
	_, axs = plt.subplots(1, 2, constrained_layout=True)
	axs = axs.flatten()
	epochs = [n for n in range(EPOCHS)]
	filenames = []

	for file in RESULTS_FILES:
		precision = []
		recall = []
		map50 = []
		map595 = []

		# get the data from .txt file
		with open(file, 'rt') as f:
			lines = [line for line in f]
		for data in lines:
			data = data.split()
			# get actual values and append them after converting: str -> float
			precision.append(float(data[P]))
			recall.append(float(data[R]))
			map50.append(float(data[MAP50]))
			map595.append(float(data[MAP]))
		# fill 'missing' values with 0s
		for i in range(len(precision), EPOCHS):
			precision.append(None)
		for i in range(len(recall), EPOCHS):
			recall.append(None)
		for i in range(len(map50), EPOCHS):
			map50.append(None)
		for i in range(len(map595), EPOCHS):
			map595.append(None)

		# get the name of the file
		filenames.append(file.split('.')[0])
		# Precision and Recall
		ax = axs[0]
		ax.plot(epochs, precision)
		ax.plot(epochs, recall)
		# ax.legend([f'Precision ({filename})', f'Recall ({filename})'], loc='center')
		ax.set_xlabel('epochs')
		ax.set_xticks(np.arange(0, EPOCHS, step=10))
		ax.grid(alpha=.15, linestyle='--')
		# mAP@.5 and mAP@.5:.95
		ax = axs[1]
		ax.plot(epochs, map50)
		ax.plot(epochs, map595)
		# ax.legend([f'mAP@.5 ({filename})', f'mAP@.5:.95 ({filename})'], loc='center')
		ax.set_xlabel('epochs')
		ax.set_xticks(np.arange(0, EPOCHS, step=10))
		ax.grid(alpha=.15, linestyle='--')

	# add legends and show final plot
	legend_1 = []
	legend_2 = []
	for i in range(len(filenames)):
		legend_1.append(f'Precision ({filenames[i]})')
		legend_1.append(f'Recall ({filenames[i]})')
		legend_2.append(f'mAP@.5 ({filenames[i]})')
		legend_2.append(f'mAP@.5:.95 ({filenames[i]})')
	axs[0].legend(legend_1, loc='center', bbox_to_anchor=(0.5, 0.35), fontsize=15)
	axs[1].legend(legend_2, loc='center', bbox_to_anchor=(0.5, 0.35), fontsize=15)
	plt.show()


if __name__ == "__main__":
	main()
