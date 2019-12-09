# Customize data sets here.
from random import shuffle
import parameters as par
import numpy as np

def _f(x):
    # return (np.sin(np.pi*x*x) + 0.1*np.random.randn(*x.shape)) * np.cos(x)
    return np.sin(2.0*np.pi*x)

# generate all x datapoints
_all_x = np.random.uniform(0*np.pi, np.pi, (1, par.num_examples)).T
np.random.shuffle(_all_x)

# partition data into different sets
(x_training, x_validation, x_test) = (_all_x[:par.training_set_size],
									  _all_x[par.training_set_size: par.training_set_size + par.validation_set_size],
									  _all_x[par.training_set_size + par.validation_set_size:])

(y_training, y_validation, y_test) = (_f(x) for x in(x_training, x_validation, x_test))



