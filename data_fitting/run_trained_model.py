#! /usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import parameters as par
import data




biases, weights = {}, {}
ckpt_reader = tf.train.load_checkpoint('./trained_model/')
weights['0'] = ckpt_reader.get_tensor('w0')
weights['1'] = ckpt_reader.get_tensor('w1')
weights['2'] = ckpt_reader.get_tensor('w2')
weights['3'] = ckpt_reader.get_tensor('w3')
weights['4'] = ckpt_reader.get_tensor('w4')
weights['5'] = ckpt_reader.get_tensor('w5')
weights['6'] = ckpt_reader.get_tensor('w6')
weights['7'] = ckpt_reader.get_tensor('w7')
biases['0'] = ckpt_reader.get_tensor('b0')
biases['1'] = ckpt_reader.get_tensor('b1')
biases['2'] = ckpt_reader.get_tensor('b2')
biases['3'] = ckpt_reader.get_tensor('b3')
biases['4'] = ckpt_reader.get_tensor('b4')
biases['5'] = ckpt_reader.get_tensor('b5')
biases['6'] = ckpt_reader.get_tensor('b6')
biases['7'] = ckpt_reader.get_tensor('b7')




def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def NN(input_layer, n):
    '''
        x: Input Tensor
        n: number of hidden layers to build
        
        Returns: output layer
        
        Builds a network starting with `input_layer`, containing n hidden
        layers, and returns `output` layer.
    '''
    def __build_hidden(inpt_layer, n):
        # Layer Number counter
        ln = 0
        layer = sigmoid(
                np.add(
                    np.matmul(
                        inpt_layer, weights['%d'%(ln)]
                    ),
                    biases['%d'%(ln)]
                )
            )
        # Increment Layer Number
        ln += 1
        while ln < n:
            layer = sigmoid(
                np.add(
                    np.matmul(
                        layer, weights['%d'%(ln)]
                    ),
                    biases['%d'%(ln)]
                )
            )
            ln += 1
        return layer        
    
    
    layer = __build_hidden(input_layer, n)
    
    output = np.add(
                np.matmul(
                    layer, weights['7']
                ),
                biases['7']
            )
    return output


output = NN(data.x_testing, par.num_of_hidden_layers)


from tensorflow.contrib.framework.python.framework import checkpoint_utils

var_list = checkpoint_utils.list_variables('./trained_model/')
for v in var_list:
    print(v)
















# tf.train.Saver(defer_build=True).restore(sess, './trained_model/trained_model.ckpt')

fig = plt.figure(figsize = par.figure_size)
plt.scatter(data.x_testing, output, color='r', s=par.marker_size)
plt.show()