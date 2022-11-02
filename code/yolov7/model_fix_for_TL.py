import torch

weights = "D:\\UNIVERSITA\\Magistrale\\SecondoAnno\\Tesi\\object_detector\\training_results\\yolov7_first_training\\weights\\best.pt"
model = torch.load(weights)

print(type(model), "\n")
for key, value in model.items():
	print(key)
print("\n", model['epoch'])
print("\n", model['training_results'])
print("\n", model['model'])
print("\n", model['best_fitness'])
print("\n", model['ema'])