# NN_RecognizingHandwrittenDigits
My first neural network to recognize hand-written digits in python with the help of this excellent book ( http://neuralnetworksanddeeplearning.com/?fbclid=IwAR01fNZ00_VPHLp-lRk2_9bGYi1HQktTmWvVxFF8haKW6NrIYa7RhQOmwzA) that helps understanding the different notions at stake when we talk about Neural Networks. We create a hole library to construct NNs specialised in recognizing the MNIST hand-written digits. 

**Repository's structure**

- The /data repository contains the MNIST data for the neural network
- The /src repository contains the source code that you need to import to create a functionnal Neural Network (there are two different versions of the network as the book guides us through some improvements of our network

**How to generate a Neural Network with the first version (for linux)**

- Open python3 in your shell
- Import mnist_loader to load the data
- Load the data with *mnist_loader.load_wrapper()* which outputs trainning_data, validation_data and test_data
- Import network
- Create your Neural Network with the command *net = network.Network(list_of_number_of_neurons_by_layer)* for instance you can run *net = network.Network([784, 30, 10])*. Note that as the MNIST images are 28x28 you have to have 784 neurons in your input layer and as we want to predict a figure your output layer should have 10 neurons.
- Train your Network with *net.stochastic_gradient_descent(training_data,number_of_epochs, mini_batch_size, learning_rate, test_data)*. You can play with these hyper-parameters to get the most accurate model as possible. A good choice can be *net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data)*.

**Improvement done on the second version**

- I changed the cost function to the log-loss functionly
- To avoid overfiting (noticeable by plotting the accuracy and the cost-function) we added early-stopping which consists in mesuring the accuracy on the validation_data to stop the fiting as soon as it caps
- We added dropout
- We improved the initialization of the weights to avoid weight saturation

**How to generate a Neural Network with the second version (for linux)**

- Open python3 in your shell
- Import mnist_loader to load the data
- Load the data with *mnist_loader.load_wrapper()* which outputs trainning_data, validation_data and test_data
- Import network2
- Create your Neural Network with the command *net = network.Network(list_of_number_of_neurons_by_layer)* for instance you can run *net = network2.Network([784, 30, 10], cost=network2.CrossEntropy)*
- Train your Network with *net.stochastic_gradient_descent(training_data,number_of_epochs, mini_batch_size, learning_rate, test_data)*. You can play with these hyper-parameters to get the most accurate model as possible. A good choice can be *net.stochastic_gradient_descent(training_data, 30, 10, 3.0, test_data)*.
