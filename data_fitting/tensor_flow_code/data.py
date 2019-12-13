# Customize data sets here.
from random import shuffle
import parameters as par
import numpy as np

x_file = open('./input_data/x.txt')
y1_file = open('./input_data/y1.txt')

x_origin = [float(x.rstrip()) for x in x_file.readlines()]
y_origin = [float(x.rstrip()) for x in y1_file.readlines()]
y_scale = max(y_origin)-min(y_origin)
y_mean = np.mean(y_origin)
y_scaled = [(y-y_mean)/y_scale for y in y_origin]

x_file.close()
y1_file.close()

xy = np.array(list(zip(x_origin, y_scaled)))
np.random.shuffle(xy)

x = np.array([[i] for i in xy[:,0]])
y = np.array([[i] for i in xy[:,1]])

(x_training, x_validation, x_test) = (x[:par.training_set_size],
									  x[par.training_set_size: par.training_set_size + par.validation_set_size],
									  x[par.training_set_size + par.validation_set_size:])

(y_training, y_validation, y_test) = (y[:par.training_set_size],
									  y[par.training_set_size: par.training_set_size + par.validation_set_size],
									  y[par.training_set_size + par.validation_set_size:])