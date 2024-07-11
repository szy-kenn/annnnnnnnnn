class Neuron:
    
    def __init__(self, weights, bias):
        self.w = weights
        self.b = bias
        self.out_soma = None
        self.out_net = None
        self.epoch = 0

    def product(arr):
        _prod = 0
        for i in arr:
            _prod *= i
        return _prod

    def relu(self, val):
        return max(0, val)
    
    def get_soma(self, inputs):
        return sum(w * i for w, i in zip(self.w, inputs)) + self.b
    
    def dE_dOut(self, target, actual):
        return actual - target
    
    # def dOut_dNet(self): 
    #     return 1 if self.out_net > 0 else 0
    
    def gradient_descent(self, old_weight, learning_rate, d_error_weight):
        return old_weight - (learning_rate * d_error_weight)
    
    def forward_pass(self, inputs):
        self.out_soma = self.get_soma(inputs)
        self.out_net = self.relu(self.out_soma)
        return self.out_net

    def backward_pass(self, inputs, output, target, net_hidden=None, net_hidden_outputs=None, *, learning_rate):
        dE_dOut = self.dE_dOut(target, output)
        

        if net_hidden != None and net_hidden_outputs != None:
            # in output neuron
            net_out = self.get_soma(net_hidden_outputs)
            out_net = self.relu(net_out)
        else:
            out_net = self.out_net
        
        dOut_dNet = 1 if out_net > 0 else 0
        dNet_dOut_Hidden = net_hidden
        dOutHidden_dNetHidden = None if net_hidden == None else 1 if net_hidden > 0 else 0

        new_weights = []
        
        # weight derivative
        for idx, dNet_dW in enumerate(inputs):
            dE_dW = dE_dOut * dOut_dNet  * dNet_dW * (dNet_dOut_Hidden if dNet_dOut_Hidden != None else 1) * (dOutHidden_dNetHidden if dOutHidden_dNetHidden != None else 1)
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

            mse = sum(0.5 * (inputs[len(self.w) - 1] - output)**2 for output, inputs in zip(outputs, all_inputs))

            print(f"Epoch: {self.epoch}, MSE: {mse},  Outputs: {outputs}")
            if (mse <= error_threshold):
                print("Done")
                print(f"Final Weights: {self.w}, Final Bias: {self.b}")
                break
            else:
                self.epoch += 1
                outputs = []

        print(f"Final Weights: {self.w}, Final Bias: {self.b}")

