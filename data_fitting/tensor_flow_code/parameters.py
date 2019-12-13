# Customize global parameters here.
num_examples = 1000
training_epochs = 3000
lambd = 1.0
validation_fraction = 0.1
test_fraction = 0.1
validation_set_size = int(num_examples*validation_fraction)
test_set_size = int(num_examples*test_fraction)
training_set_size = num_examples - validation_set_size - test_set_size
training_epochs_digits = len(list(str(training_epochs)))
display_step = int(training_epochs*0.1)
PLOT_FIGURE = True
figure_size = (8, 4.5)
marker_size = 5.0
neuron_of_layer = [1, 50, 50, 50, 50, 50, 50, 50, 1]
num_of_all_layers = len(neuron_of_layer)
num_of_hidden_layers = num_of_all_layers - 2

assert type(num_examples)==int and num_examples>0
assert training_set_size>0
assert type(training_epochs)==int and training_epochs>10
for i in neuron_of_layer:
	assert type(i)==int and i>0
