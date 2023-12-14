import numpy as np
import pandas as pd
import sys


class NeuralNetwork:
    def __init__(self):
        self.print_dict = {}
        self.dict_keys = ['a', 'b', 'h1', 'h2', 'h3', 'o', 't', 'delta_h1', 'delta_h2', 'delta_h3', 'delta_o', 'w_bias_h1', 'w_a_h1', 'w_b_h1', 
                          'w_bias_h2', 'w_a_h2', 'w_b_h2', 'w_bias_h3', 'w_a_h3', 'w_b_h3', 'w_bias_o', 'w_h1_o', 'w_h2_o', 'w_h3_o']
        self.print_dict.update({key: "-" for key in self.dict_keys})

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def initialize_weights(self):
        """Initialize weights according the to given weights"""
        # weights (input to hidden) related to a hidden unit which will be on the row of the matrix placed in different columns of that row
        # 3x3
        weights_input_hidden = np.array([[ 0.2, -0.3, 0.4],
                                         [-0.5, -0.1, -0.4],
                                         [ 0.3, 0.2, 0.1]])
        # weights (hidden to output) related to a output unit which will be on the row of the matrix placed in different columns of that row
        weights_hidden_output = np.array([[-0.1, 0.1, 0.3, -0.4]]) # 1x4
        
        # Update weights in parameter dictionary
        flattened_weights = weights_input_hidden.flatten()
        keys = ['w_bias_h1', 'w_a_h1', 'w_b_h1', 'w_bias_h2', 'w_a_h2', 'w_b_h2', 'w_bias_h3', 'w_a_h3', 'w_b_h3']
        self.print_dict.update({key: value for key, value in zip(keys, flattened_weights)})
        flattened_weights = weights_hidden_output.flatten()
        keys = ['w_bias_o', 'w_h1_o', 'w_h2_o', 'w_h3_o']
        self.print_dict.update({key: value for key, value in zip(keys, flattened_weights)})

        return weights_input_hidden, weights_hidden_output

    def forward_propagation(self, inputs, weights_input_hidden, weights_hidden_output):
        self.print_dict.update({key: value for key, value in zip(["a", "b"], inputs.flatten())})
        
        inputs = np.insert(inputs, 0, 1) # bias input for hidden layer
        
        # ------------- Propogation from input to hidden layer ----------------
        # output of each hidden unit will be on the each row of the matrix [hx1]
        net_hidden = np.dot(weights_input_hidden, inputs) # shape - number of hidden units x 1
        output_hidden = self.sigmoid(net_hidden)
        self.print_dict.update({key: value for key, value in zip(["h1", "h2", "h3"], output_hidden.flatten())})

        output_hidden = np.insert(output_hidden, 0, 1) # bias input for output layer
        
        # ------------- Propagation from hidden to output layer ------------------
        net_output = np.dot(weights_hidden_output, output_hidden) # 1x4 - 4x1 = 1x1
        output = self.sigmoid(net_output) # shape - no. of output units x 1
        self.print_dict.update({key: value for key, value in zip(["o"], output.flatten())})

        return inputs, output_hidden, output

    def backward_propagation(self, inputs, output_hidden, output, target, weights_input_hidden, weights_hidden_output, eta):
        self.print_dict.update({key: value for key, value in zip(["t"], target.flatten())})
        
        # ------------ error of output layer ----------------------
        output_error = (target - output)
        output_delta = output_error * self.sigmoid_derivative(output) # shape - no of output units x 1
        
        # ------------ error of hidden layers ----------------------
        # ((no. of hidden units + 1) x no. of output units ) x (no of output units x 1)
        # result shape = (no. of hidden units + 1) x 1
        hidden_error = np.dot(weights_hidden_output.T, output_delta)
        # hidden delta = 3x1
        hidden_delta = hidden_error[1:] * self.sigmoid_derivative(output_hidden[1:])
        self.print_dict.update({key: value for key, value in zip(['delta_h1', 'delta_h2', 'delta_h3'], hidden_delta.flatten())})
        self.print_dict.update({key: value for key, value in zip(["delta_o"], output_delta.flatten())})

        # ------------- Weight Update --------------------
        weights_hidden_output[:, 0] += eta * output_delta # bias update
        # Outer product for output = (1x1) x (3x1) = (3x1)
        weights_hidden_output[:, 1:] += eta * np.outer(output_delta, output_hidden[1:])
        # Outer product for input-hidden = (3x1) x (3x1) = (3x3)
        weights_input_hidden[:, 0] += eta * hidden_delta # bias update
        weights_input_hidden[:, 1:] += eta * np.outer(hidden_delta, inputs[1:])
        
        flattened_weights = weights_input_hidden.flatten()
        keys = ['w_bias_h1', 'w_a_h1', 'w_b_h1', 'w_bias_h2', 'w_a_h2', 'w_b_h2', 'w_bias_h3', 'w_a_h3', 'w_b_h3']
        self.print_dict.update({key: value for key, value in zip(keys, flattened_weights)})
        flattened_weights = weights_hidden_output.flatten()
        keys = ['w_bias_o', 'w_h1_o', 'w_h2_o', 'w_h3_o']
        self.print_dict.update({key: value for key, value in zip(keys, flattened_weights)})

        return weights_input_hidden, weights_hidden_output

    def train_neural_network(self, data, eta, iterations):
        inputs = data.iloc[:, :-1].values
        targets = data.iloc[:, -1].values
        
        input_size = inputs.shape[1]
        hidden_size = 3
        output_size = 1
        
        weights_input_hidden, weights_hidden_output = self.initialize_weights()
        for key in self.dict_keys : print(f"{self.print_dict[key]}", end=",")
        print()

        for epoch in range(iterations):
            for i in range(len(inputs)):
                input_data, output_hidden, output = self.forward_propagation(inputs[i], weights_input_hidden, weights_hidden_output)
                weights_input_hidden, weights_hidden_output = self.backward_propagation(input_data, 
                                                                                        output_hidden,
                                                                                        output, 
                                                                                        targets[i],
                                                                                        weights_input_hidden,
                                                                                        weights_hidden_output,
                                                                                        eta)
                for key in self.dict_keys : print(f"{self.print_dict[key]}", end=",")
                print()

if __name__ == "__main__":
    data_path = sys.argv[2]
    eta = float(sys.argv[4])
    iterations = int(sys.argv[6])
    data = pd.read_csv(data_path, header=None)
    NeuralNetwork().train_neural_network(data, eta, iterations)
