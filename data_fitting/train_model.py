#! /usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import os
import data
import parameters as par



figure_path = os.getcwd() + '/figure'

try:
    os.mkdir("figure")
except:
    pass


tf.reset_default_graph()




X = tf.placeholder(tf.float32, [None, par.n_input], name='X')
Y = tf.placeholder(tf.float32, [None, par.n_output], name='Y')

layer_units = [1, 50, 50, 50, 50, 50, 50, 50, 1]


weights = {str(i): tf.Variable(tf.random_normal([layer_units[i], layer_units[i+1]]), name = 'w{:d}'.format(i))
            for i in range(len(layer_units)-1)}

biases = {str(i): tf.Variable(tf.random_normal([layer_units[i+1]]), name = 'b{:d}'.format(i))
            for i in range(len(layer_units)-1)}





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
        layer = tf.nn.sigmoid(
                tf.add(
                    tf.matmul(
                        inpt_layer, weights['%d'%(ln)],
                        name='layer%d'%(ln)
                    ),
                    biases['%d'%(ln)]
                )
            )
        # Increment Layer Number
        ln += 1
        while ln < n:
            layer = tf.nn.sigmoid(
                tf.add(
                    tf.matmul(
                        layer, weights['%d'%(ln)],
                        name='layer%d'%(ln)
                    ),
                    biases['%d'%(ln)]
                )
            )
            ln += 1
        return layer        
    
    
    layer = __build_hidden(input_layer, n)
    
    output = tf.add(
                tf.matmul(
                    layer, weights[str(len(layer_units)-2)],
                    name="out"
                ),
                biases[str(len(layer_units)-2)]
            )
    return output


output = NN(X, par.num_of_hidden_layers)
cost = tf.reduce_mean(tf.square(output - Y))
optimizer = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(output - Y))






sess = tf.InteractiveSession()

costs = []
epoches = []
total_error = []


init = tf.global_variables_initializer()
sess.run(init)





if par.PLOT_FIGURE:
    fig = plt.figure(figsize = par.figure_size)
    plt.scatter(data.x_training, data.y_training, c='b', label='train', s=par.marker_size)
    plt.scatter(data.x_testing, data.y_testing, c='r', label='validation', s=par.marker_size)
    plt.legend()
    fig.savefig(os.path.join(figure_path, 'training_test_set.jpg'))   




for epoch in range(1, par.training_epochs+1):


    _, c = sess.run(
        [ optimizer, cost ],
        feed_dict={
            X: data.x_training,
            Y: data.y_training
        }
    )

    costs.append(c)
    epoches.append(epoch)   
    mse = sess.run(tf.nn.l2_loss(output - data.y_testing),  feed_dict={X: data.x_testing})
    total_error.append(mse)    
    
    if not epoch % par.display_step :
        print('Epoch: %03d\tCost: %.9f'%(epoch, c))

        if par.PLOT_FIGURE:
            # plt.clf()
            fig = plt.figure(figsize = par.figure_size)
            plt.scatter(data.x_testing, data.y_testing, color='b', s=par.marker_size)
            plt.scatter(data.x_testing, sess.run(output, feed_dict={X: data.x_testing}), color='r', s=par.marker_size)

            
            fig.savefig(os.path.join(figure_path, '%4d_training_plot.jpg'%(epoch)))




save_model_list = [weights[i] for i in weights] + [biases[i] for i in biases]


saver = tf.train.Saver(save_model_list)
saver.save(sess, './trained_model/trained_model.ckpt')


    
print('Epoch: %03d\tMSE: %.4e'%(epoch, c))
print('Training complete!')



plt.close('all')
sess.close()











