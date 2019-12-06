#! /usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import parameters as par
import data
from tensorflow.contrib.framework.python.framework import checkpoint_utils as cu

def _sigmoid(x):
	return 1/(1 + np.exp(-x))

def _DNN(input_data, num_of_hidden_layers):
    def _build_hidden(num_of_hidden_layers):
        current_layer = 0
        output_current_layer = input_data[:]
        while current_layer < num_of_hidden_layers:
            Z = np.add(
                    np.matmul(output_current_layer, _weights['%d'%(current_layer)]),
                    _biases['%d'%(current_layer)]
                )
            output_current_layer = _sigmoid(Z)
            current_layer += 1
        return str(current_layer), output_current_layer 
        
    next_layer, layer = _build_hidden(num_of_hidden_layers)  
    output_NN = np.add(np.matmul(layer, _weights[next_layer]), _biases[next_layer])
    return output_NN

_ckpt_reader = tf.train.load_checkpoint('./trained_model/')

(_weights, _biases) = ({str(i): _ckpt_reader.get_tensor('w{:d}'.format(i)) for i in range(8)}, 
                     {str(i): _ckpt_reader.get_tensor('b{:d}'.format(i)) for i in range(8)})

_output = _DNN(data.x_test, par.num_of_hidden_layers)

_var_list = cu.list_variables('./trained_model/')

for i in _var_list:
    print(i)

_fig = plt.figure(figsize = par.figure_size)
plt.scatter(data.x_test, data.y_test, color='b', label='test set', s=par.marker_size)
plt.scatter(data.x_test, _output, color='g', label='DNN prediction', s=par.marker_size)
plt.legend()
plt.show()
plt.close()