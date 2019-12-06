#! /usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import os
import data
import parameters as par

try:
    os.mkdir("./figure")
except:
    pass

_figure_path = os.getcwd() + '/figure'

tf.reset_default_graph()

_X = tf.placeholder(tf.float32, [None, par.neuron_of_layer[0]], name='Input')
_Y = tf.placeholder(tf.float32, [None, par.neuron_of_layer[-1]], name='Output')
_weights = {str(i): tf.Variable(tf.random_normal([par.neuron_of_layer[i], par.neuron_of_layer[i+1]]),
            name = 'w{:d}'.format(i)) for i in range(par.num_of_all_layers-1)}
_biases = {str(i): tf.Variable(tf.zeros([par.neuron_of_layer[i+1]]), name = 'b{:d}'.format(i))
            for i in range(par.num_of_all_layers-1)}

def _DNN(input_data, num_of_hidden_layers):
    def _build_hidden(num_of_hidden_layers):
        current_layer = 0
        output_current_layer = input_data[:]
        while current_layer < num_of_hidden_layers:
            Z = tf.add(
                    tf.matmul(output_current_layer, _weights['%d'%(current_layer)]),
                    _biases['%d'%(current_layer)]
                )
            output_current_layer = tf.nn.sigmoid(Z, name='layer%d'%(current_layer))
            current_layer += 1
        return str(current_layer), output_current_layer 

    next_layer, layer = _build_hidden(num_of_hidden_layers)  
    output_NN = tf.add(tf.matmul(layer, _weights[next_layer]), _biases[next_layer])
    return output_NN

_output = _DNN(_X, par.num_of_hidden_layers)
_cost = tf.reduce_mean(tf.square(_output - _Y))
_optimizer = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(_output - _Y))

_sess = tf.InteractiveSession()
_init = tf.global_variables_initializer()
_sess.run(_init)

if par.PLOT_FIGURE:
    fig = plt.figure(figsize=par.figure_size)
    plt.scatter(data.x_training, data.y_training, c='r', label='training set', s=par.marker_size)
    plt.scatter(data.x_validation, data.y_validation, c='b', label='validation set', s=par.marker_size)
    plt.legend()
    fig.savefig(os.path.join(_figure_path, 'training_and_validation_sets.jpg'))
   
for epoch in range(1, par.training_epochs+1):
    _, c = _sess.run(
        [_optimizer, _cost ],
        feed_dict={_X: data.x_training, _Y: data.y_training}
    )

    mse = _sess.run(tf.nn.l2_loss(_output - data.y_validation), feed_dict={_X: data.x_validation})
    
    if not epoch % par.display_step :
        print('Epoch: %0*d\tCost : %.6f'%(par.training_epochs_digits, epoch, c))
        if par.PLOT_FIGURE:
            fig = plt.figure(figsize=par.figure_size)
            plt.scatter(data.x_validation, data.y_validation, color='b', label='validation set', s=par.marker_size)
            plt.scatter(data.x_validation, _sess.run(_output, feed_dict={_X: data.x_validation}),
                        color='r', label='DNN prediction', s=par.marker_size) 
            plt.legend()         
            fig.savefig(os.path.join(_figure_path, '%*d_DNN_prediction.jpg'%(par.training_epochs_digits, epoch)))

print('Epoch: %03d\tMSE  : %.6f'%(epoch, c), 'Training complete!', sep='\n')

_save_model_list = [_weights[i] for i in _weights] + [_biases[i] for i in _biases]
_saver = tf.train.Saver(_save_model_list)
_saver.save(_sess, './trained_model/trained_model.ckpt')

plt.close('all')
_sess.close()











