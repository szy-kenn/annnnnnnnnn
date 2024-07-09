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


        for idx in range(1, self.neuron_count):

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

    def backward_pass(self, inputs, output, target, neurons, learning_rate):

        new_ws = []
        new_bs = []

        for neuron in neurons:
            new_weights, bias = neuron.backward_pass(inputs, output, target, learning_rate)
            neuron.w = new_weights
            neuron.b = bias
            
            if len(neurons):
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
                final_ws.clear()
                final_bs.clear()
                net_outputs = self.forward_pass(inputs, self.hidden_layer_neurons)
                [output] = self.forward_pass(net_outputs, [self.output_neuron])
                outputs.append(output)

                # if (outputs.count(0)):
                    # print("Vanishing gradient. Stopping.")
                    # epoch = max_epoch

                net_ws, net_bs = self.backward_pass(net_outputs, output, target, [self.output_neuron], learning_rate)
                final_ws.append(net_ws)
                final_bs.append(net_bs)

                for net, neuron in zip(net_outputs, self.hidden_layer_neurons):
                    out_ws, out_b = self.backward_pass(inputs, net, target, [neuron], learning_rate)
                    final_ws.append(out_ws)
                    final_bs.append(out_b)

            mse = sum(0.5 * (output - target)**2 for output, target in zip(outputs, all_targets))

            print(f"Epoch: {epoch}, MSE: {mse},  Outputs: {outputs}")
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