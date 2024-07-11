from math import sqrt
import random
from Neuron import Neuron 

class ANN: 
    def __init__(self, neuron_count, fan_in, fan_out, weights=None, biases=None):
        self.neuron_count = neuron_count
        self.fan_in = fan_in
        self.fan_out = fan_out
        
        self.hidden_layer_neurons = []

        if biases != None:
            self.output_neuron = Neuron(weights=weights[0], bias=biases[0])
        else:
            self.output_neuron = Neuron(weights=ANN.get_random_weights(self.neuron_count), bias=ANN.get_random_bias(self.neuron_count))

        for idx in range(1, self.neuron_count+1):

            if weights != None:
                self.hidden_layer_neurons.append(Neuron(weights=weights[idx], bias=biases[idx]))    
            else:
                self.hidden_layer_neurons.append(Neuron(weights=ANN.get_random_weights(self.fan_in), bias=ANN.get_random_bias(self.neuron_count)))    

    @classmethod
    def get_random_weights(cls, input_count):
        limit = sqrt(6) / sqrt(input_count)
        weights = []
        # weights and biases
        for _ in range(input_count):
            weights.append(random.uniform(-limit, +limit))
        
        return weights

    @classmethod
    def get_random_bias(cls, input_count):
        min_value = -sqrt(6) / sqrt(input_count)
        max_value = sqrt(6) / sqrt(input_count)
        b = random.uniform(min_value, max_value)
        return b

    def forward_pass(self, inputs, neurons):
        outputs = []
        for neuron in neurons:
            outputs.append(neuron.forward_pass(inputs))

        return outputs

    def backward_pass(self, inputs, output, target, net_hidden=None, net_hidden_outputs=None, *, neurons, learning_rate):

        new_ws = []
        new_bs = []

        for neuron in neurons:
            new_weights, bias = neuron.backward_pass(inputs, output, target, net_hidden, net_hidden_outputs, learning_rate=learning_rate)
            neuron.w = new_weights
            neuron.b = bias
            
            if len(neurons) == 1:
                new_ws = new_weights
                new_bs = bias
            else:
                new_ws.append(new_weights)
                new_bs.append(bias)

        return new_ws, new_bs

    def train(self, all_inputs, all_targets, learning_rate, max_epoch, error_tolerance):

        epoch = 0

        while epoch < max_epoch:           
        # for _ in range(1):           
            outputs = []
            final_ws = []
            final_bs = []

            for inputs, target in zip(all_inputs, all_targets):
                final_ws = []
                final_bs = []
                net_hidden_outputs = self.forward_pass(inputs, self.hidden_layer_neurons)
                [output] = self.forward_pass(net_hidden_outputs, [self.output_neuron])
                outputs.append(output)

                net_ws, net_bs = self.backward_pass(net_hidden_outputs, output, target, neurons=[self.output_neuron], learning_rate=learning_rate)
                final_ws.append(net_ws)
                final_bs.append(net_bs)

                for net_hidden, neuron in zip(net_hidden_outputs, self.hidden_layer_neurons):
                    out_ws, out_b = self.backward_pass(inputs, output, target, net_hidden, net_hidden_outputs, neurons=[neuron], learning_rate=learning_rate)
                    final_ws.append(out_ws)
                    final_bs.append(out_b)

            mse = sum(0.5 * (target - output)**2 for output, target in zip(outputs, all_targets))

            if epoch % 1000 == 0:
                print(f"Epoch: {epoch} \t MSE: {mse}")
                # print(f"Epoch: {epoch} \t MSE: {mse} \nOutputs: {outputs}")
            # print(f"Epoch: {epoch}")
            # print(f"Epoch: {epoch}, MSE: {mse},  Outputs: {outputs}")
            # print(f"Final Weights: {final_ws}\nFinal Bias: {final_bs}")

            if (mse <= error_tolerance):
                print("Done")
                break
            else:
                epoch += 1
    
        print(f"Epoch: {epoch}, MSE: {mse},  Outputs: {outputs}")

        print(f"Final Weights: {final_ws}\nFinal Bias: {final_bs}")
        return final_ws, final_bs
        # print(len(final_ws[2]), len(final_bs[2]))

    def test(self, val: list):
        if len(val) != self.fan_in:
            raise ValueError("Invalid input size")
        
        net_outputs = self.forward_pass(val, self.hidden_layer_neurons)
        [output] = self.forward_pass(net_outputs, [self.output_neuron])
        return output