#! /usr/bin/env python
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from random import shuffle
import os



figure_path = os.getcwd() + '/figure'

try:
    os.mkdir("figure")
except:
    pass


# global parameters
num_examples = 5000
test_fraction = 0.75
train_size = int(num_examples*test_fraction)



training_epochs = 3500
display_step = training_epochs*0.1
PLOT_FIGURE = True




n_input = 1
hidden_nodes = 50
n_hidden_0 = hidden_nodes
n_hidden_1 = hidden_nodes
n_hidden_2 = hidden_nodes
n_hidden_3 = hidden_nodes
n_hidden_4 = hidden_nodes
n_hidden_5 = hidden_nodes
n_hidden_6 = hidden_nodes

num_of_hidden_layers = 7
n_output = n_input




# test function

def f(x):
    return np.sin(2.0*np.pi*x*x/1.5) + 0.1*np.random.randn(*x.shape)


# generate all x datapoints
all_x = np.random.uniform(0*np.pi, np.pi, (1, num_examples)).T
np.random.shuffle(all_x)




# partition data into different sets
x_training = all_x[:train_size]
x_testing = all_x[train_size:]
y_training = f(x_training)
y_testing = f(x_testing)

if PLOT_FIGURE:
    fig = plt.figure(figsize=(8,4.5))
    plt.scatter(x_training, y_training, c='b', label='train', s=0.5)
    plt.scatter(x_testing, y_testing, c='r', label='validation', s=0.5)
    plt.legend()
    fig.savefig(os.path.join(figure_path, 'training_test_set.jpg'))   











tf.reset_default_graph()


X = tf.placeholder(tf.float32, [None, n_input], name='X')
Y = tf.placeholder(tf.float32, [None, n_output], name='Y')


weights = {
    'h0': tf.Variable(tf.random_normal([n_input, n_hidden_0])),
    'h1': tf.Variable(tf.random_normal([n_hidden_0, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_5, n_hidden_6])),
    'out': tf.Variable(tf.random_normal([n_hidden_6, n_output])),
}

biases = {
    'h0': tf.Variable(tf.random_normal([n_hidden_0])),
    'h1': tf.Variable(tf.random_normal([n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_2])),
    'h3': tf.Variable(tf.random_normal([n_hidden_3])),
    'h4': tf.Variable(tf.random_normal([n_hidden_4])),
    'h5': tf.Variable(tf.random_normal([n_hidden_5])),
    'h6': tf.Variable(tf.random_normal([n_hidden_6])),
    'out': tf.Variable(tf.random_normal([n_output])),
}

parameter = {'weights': weights, 'biases': biases}




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
                        inpt_layer, weights['h%d'%(ln)],
                        name='layer%d'%(ln)
                    ),
                    biases['h%d'%(ln)]
                )
            )
        # Increment Layer Number
        ln += 1
        while ln < n:
            layer = tf.nn.sigmoid(
                tf.add(
                    tf.matmul(
                        layer, weights['h%d'%(ln)],
                        name='layer%d'%(ln)
                    ),
                    biases['h%d'%(ln)]
                )
            )
            ln += 1
        return layer        
    
    
    layer = __build_hidden(input_layer, n)
    
    output = tf.add(
                tf.matmul(
                    layer, weights['out'],
                    name="out"
                ),
                biases['out']
            )
    return output


output = NN(X, num_of_hidden_layers)
cost = tf.reduce_mean(tf.square(output - Y))
optimizer = tf.train.AdamOptimizer().minimize(tf.nn.l2_loss(output - Y))


init = tf.global_variables_initializer()







sess = tf.Session()

costs = []
epoches = []
total_error = []


sess.run(init)




for epoch in range(training_epochs):


    _, c = sess.run(
        [ optimizer, cost ],
        feed_dict={
            X:x_training,
            Y:y_training
        }
    )

    costs.append(c)
    epoches.append(epoch)
    
    mse = sess.run(tf.nn.l2_loss(output - y_testing),  feed_dict={X:x_testing})
    total_error.append(mse)    
    
    if epoch%display_step == 0:
        print('Epoch: %03d\tCost: %.9f'%(epoch, c))

        if PLOT_FIGURE:
            # plt.clf()
            fig = plt.figure(figsize=(20,8))
            plt.scatter( x_testing, y_testing, color='g')
            plt.scatter( x_testing, sess.run(output, feed_dict={X: x_testing}), color='r')

            
            fig.savefig(os.path.join(figure_path, '%4d_training_plot.jpg'%(epoch)))

    
print('Epoch: %03d\tMSE: %.4e'%(epoch+1, c))
print('Training complete!')


saver_weights = tf.train.Saver(weights)
saver_weights.save(sess, './trained_model/trained_weights.ckpt')
saver_biases = tf.train.Saver(biases)
saver_biases.save(sess, './trained_model/trained_biases.ckpt')


fig = plt.figure(figsize=(20,8))
plt.scatter( x_testing, y_testing, color='g')
plt.scatter( x_testing, sess.run(output, feed_dict={X: x_testing}), color='r')
# display.display(plt.gcf())
# display.clear_output(wait=True)
# plt.show()




