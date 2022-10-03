import matplotlib.pyplot as plt
import numpy as np


# location of the file 'results.txt'
RESULTS_FILE = 'D:\\UNIVERSITA\\Magistrale\\SecondoAnno\\Tesi\\object_detector\\training_results\\yolov7_025\\results.txt'
# these are the indices corresponding to the order of the statistichs in the 'results.txt' file
P = 8             # Precision
R = 9             # Recall
MAP50 = 10        # mAP@.5
MAP = 11          # mAP@0.5:.95
BOXLOSS_VAL = 12  # Validation Box Loss 
OBJLOSS_VAL = 13  # Validation Objectness Loss
CLSLOSS_VAL = 14  # Validation Classification Loss
BOXLOSS_TRAIN = 2 # Training Box Loss 
OBJLOSS_TRAIN = 3 # Training Objectness Loss
CLSLOSS_TRAIN = 4 # Training Classification Loss


def main():

	precision = []
	recall = []
	map50 = []
	map595 = []
	boxloss_val = []
	objloss_val = []
	clsloss_val = []
	boxloss_train = []
	objloss_train = []
	clsloss_train = []

	# get the data from .txt file
	with open(RESULTS_FILE, 'rt') as f:
		lines = [line for line in f]
	for data in lines:
		data = data.split()
		# get actual values and append them after converting: str -> float
		precision.append(float(data[P]))
		recall.append(float(data[R]))
		map50.append(float(data[MAP50]))
		map595.append(float(data[MAP]))
		boxloss_val.append(float(data[BOXLOSS_VAL]))
		objloss_val.append(float(data[OBJLOSS_VAL]))
		clsloss_val.append(float(data[CLSLOSS_VAL]))
		boxloss_train.append(float(data[BOXLOSS_TRAIN]))
		objloss_train.append(float(data[OBJLOSS_TRAIN]))
		clsloss_train.append(float(data[CLSLOSS_TRAIN]))

	# plot the data
	epochs = [n for n in range(len(lines))]
	n_epochs = len(epochs)
	_, axs = plt.subplots(2, 2, constrained_layout=True)
	axs = axs.flatten()
	# precision - recall - mAP@.5 - mAP@.5:.95
	ax = axs[0]
	ax.plot(epochs, precision, '-b')
	ax.plot(epochs, recall, '-g')
	ax.plot(epochs, map50)
	ax.plot(epochs, map595)
	ax.legend(['Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95'], loc='center')
	ax.set_xlabel('epochs')
	ax.set_xticks(np.arange(0, n_epochs, step=10))
	ax.grid(alpha=.15, linestyle='--')
	# box loss
	ax = axs[1]
	ax.plot(epochs, boxloss_val)
	ax.plot(epochs, boxloss_train)
	ax.legend(['box_loss val', 'box_loss train'], loc='center')
	ax.set_xlabel('epochs')
	ax.set_xticks(np.arange(0, n_epochs, step=10))
	ax.grid(alpha=.15, linestyle='--')
	# objectness loss
	ax = axs[2]
	ax.plot(epochs, objloss_val)
	ax.plot(epochs, objloss_train)
	ax.legend(['obj_loss val', 'obj_loss train'], loc='center')
	ax.set_xlabel('epochs')
	ax.set_xticks(np.arange(0, n_epochs, step=10))
	ax.grid(alpha=.15, linestyle='--')
	# classification loss
	ax = axs[3]
	ax.plot(epochs, clsloss_val)
	ax.plot(epochs, clsloss_train)
	ax.legend(['cls_loss val', 'cls_loss train'], loc='center')
	ax.set_xlabel('epochs')
	ax.set_xticks(np.arange(0, n_epochs, step=10))
	ax.grid(alpha=.15, linestyle='--')
	# show final plot
	plt.show()


if __name__ == "__main__":
	main()
