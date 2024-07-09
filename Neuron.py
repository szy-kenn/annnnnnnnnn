class Neuron:
    
    def __init__(self, weights, bias):
        self.w = weights
        self.b = bias
        self.out_soma = None
        self.out_net = None
        self.epoch = 0

    def relu(self, val):
        return max(0, val)
    
    def get_soma(self, inputs):
        return sum(w * i for w, i in zip(self.w, inputs)) + self.b
    
    def dE_dOut(self, target, actual):
        return 2 * (actual - target)
    
    def dOut_dNet(self): 
        return 1 if self.out_net > 0 else 0
    
    def gradient_descent(self, old_weight, learning_rate, d_error_weight):
        return old_weight - (learning_rate * d_error_weight)
    
    def forward_pass(self, inputs):
        self.out_soma = self.get_soma(inputs)
        self.out_net = self.relu(self.out_soma)
        return self.out_net
        # print(self.out_soma, self.out_net)

    def backward_pass(self, inputs, output, target, learning_rate):
        dE_dOut = self.dE_dOut(target, output)
        dOut_dNet = self.dOut_dNet()

        new_weights = []
        
        # weight derivative
        for idx, dNet_dW in enumerate(inputs):
            dE_dW = dNet_dW * dE_dOut * dOut_dNet
            new_weight = self.gradient_descent(self.w[idx], learning_rate, dE_dW)
            new_weights.append(new_weight)

        dE_dB = dE_dOut * dOut_dNet
        new_bias = self.gradient_descent(self.b, learning_rate, dE_dB)
        return new_weights, new_bias

    def train(self, all_inputs, max_epoch, learning_rate, error_threshold):

        outputs = []

        for _ in range(max_epoch):

            for inputs in all_inputs:
                output = self.forward_pass(inputs)
                outputs.append(output)
                new_weights, new_bias = self.backward_pass(inputs, output, learning_rate)
                self.w = new_weights
                self.b = new_bias

            mse = sum(0.5 * (output * inputs[len(self.w) - 1])**2 for output, inputs in zip(outputs, all_inputs))

            print(f"Epoch: {self.epoch}, MSE: {mse},  Outputs: {outputs}")
            if (mse <= error_threshold):
                print("Done")
                print(f"Final Weights: {self.w}, Final Bias: {self.b}")
                break
            else:
                self.epoch += 1
                outputs = []

        print(f"Final Weights: {self.w}, Final Bias: {self.b}")

