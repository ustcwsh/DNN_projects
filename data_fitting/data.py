from random import shuffle
import parameters as par
import numpy as np



def f(x):
    return np.sin(2.0*np.pi*x*x/1.5) + 0.1*np.random.randn(*x.shape)
    # return np.sin(2.0*np.pi*x)


# generate all x datapoints
all_x = np.random.uniform(0*np.pi, np.pi, (1, par.num_examples)).T
np.random.shuffle(all_x)




# partition data into different sets
x_training = all_x[:par.train_size]
x_testing = all_x[par.train_size:]
y_training = f(x_training)
y_testing = f(x_testing)


