import numpy as np


class Layer:
    def __init__(self, nIn: int, nOut: int) -> None:
        self.in_size = nIn
        self.out_size = nOut

        self.weights = np.random.uniform(-1, 1, (nOut, nIn))
        self.biases = np.random.uniform(-1, 1, (nOut, 1))

        # Gradient Arrays
        self.weight_grads = np.zeros_like(self.weights)
        self.bias_grads = np.zeros_like(self.biases)

        # These two are necessary when doing backpropagation.
        self.input_set_list = []
        self.output_set_list = []

    def _forward_pass(self, inputs: np.ndarray):
        raw_outputs = np.matmul(self.weights, inputs) + \
            self.biases  # RawOutputs

        # Passing through activation function to squash values between 0 and 1.
        outputs = np.tanh(raw_outputs)

        self.input_set_list.append(inputs)
        self.output_set_list.append(outputs)

        return outputs

    def _backward_pass(self, out_grads: np.ndarray, sample_index: int):
        # ["dl/dOut"s]=out_grads, dOut/dRW = 1-Out**2 | tanh(RW)=Out
        dL_over_dRW = (1 - self.output_set_list[sample_index]**2) * out_grads

        # dL/dRW * dRW/dwij = dL/dwij. We do this for all weights at once by matrix multiplication.
        self.weight_grads += np.matmul(dL_over_dRW, np.transpose(self.input_set_list[sample_index]))

        # Bias grads: dL/db = dL/dRW * dRW/db = dL/db | dRW/db = 1
        self.bias_grads += dL_over_dRW

        # Layer's inputs' grads. 
        input_grads = np.matmul(np.transpose(self.weights), dL_over_dRW)
        return input_grads

    def _optimize_params(self, bump: float):
        self.weights += -self.weight_grads * bump
        self.biases += -self.bias_grads * bump

    def _set_grads_zero(self):
        #Sets all parameters' gradients to zero. Paramaters = Weights and Biases
        self.weight_grads = np.zeros_like(self.weights)
        self.bias_grads = np.zeros_like(self.biases)

    def __repr__(self) -> str:
        return f"Layer: WeightsShape->{self.weights.shape}"


class MLP:  # Multi_Layer_Perceptron
    def __init__(self, inputs_size: int, hidden_layer_sizes: list[int], ) -> None:
        self.layers: list[Layer] = self._create_layers(
            inputs_size, hidden_layer_sizes)
        self.last_loss = 0

    def _create_layers(self, input_size: int, hidden_layer_sizes: list[int]):
        layer_sizes = [input_size] + hidden_layer_sizes + [1]
        layers = []

        for ls in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[ls], layer_sizes[ls + 1])
            layers.append(layer)

        return layers

    def _forward_pass_one(self, inputs: np.ndarray):
        # Check size compatiblity.
        if self.layers[0].in_size != len(inputs[0]):
            raise ValueError(
                "'inputs's size must be compatible with first layers input size.")

        current_inputs = np.transpose(inputs)  # Keeps current layer inputs.

        for l in self.layers:
            # We pass previous layer's outputs to next layer's inputs.
            current_inputs = l._forward_pass(current_inputs)

        return current_inputs  # Last output of forward pass.

    def _calc_loss(self, preds: list, gts: np.ndarray):
        # Calculates loss and derivatives of preds.
        loss = 0
        pred_grads = []  # "dL/dpred" List

        for p, gt in zip(preds, gts):
            loss += (p[0] - gt)**2 # "dL/ddiff * ddiff/dp" where diff = (p -gt)
            pred_grads.append(2*(p-gt) * 1)

        self.last_loss = loss
        return (preds, pred_grads, loss)

    def forward_pass_all(self, dataset: np.ndarray, gts: np.ndarray):
        # Forward passes all dataset samples.
        preds = []
        for sample in dataset:
            preds.append(self._forward_pass_one(sample))
        
        return self._calc_loss(preds, gts)

    def backward_pass(self, pred_grads: list):

        for i in range(len(pred_grads)):
            out_grads = pred_grads[i]
            
            # We need to go backwards through layers.
            for l in reversed(self.layers):
                out_grads = l._backward_pass(out_grads, i)

    def gradient_descent_optimization(self, bump: float):
        #Applies gradient descent optimization to all parameters of network.
        for l in self.layers:
            l._optimize_params(bump) 
            l._set_grads_zero()

    def train_loop(self, dataset: np.ndarray, gts: np.ndarray, gdo_bump: float, loop_count:int = 1):
        #Applies "forward-backward-gdo" loop 'loop_count' amount of times. 
        for i in range(loop_count):
            result = self.forward_pass_all(dataset, gts)
            self.backward_pass(result[1])
            self.gradient_descent_optimization(gdo_bump)
       
        #Results
        print("loss: ", self.last_loss)
        print("last preds: ", result[0])

    def print_params(self):
        #Prints all weights' and biases' gradients.
        for l in self.layers:
            print(l)
            print("Weight grads:\n", l.weight_grads)
            print("bias grads:\n", l.bias_grads)

    


# input_layer = np.transpose(np.array([[1, 2, 3]]))
# hidden_1 = Layer(3, 4)
# print("weights:\n", hidden_1.weights)
# print("weight_grads:\n", hidden_1.weight_grads)
# print()
# print("biases:\n", hidden_1.biases)
# print("bias_grads:\n", hidden_1.bias_grads)
# print()
# outputs = hidden_1._forward_pass(inputs=input_layer)
# print("outputs:", outputs)
# print("input list:", hidden_1.input_set_list)
# print("output list:", hidden_1.output_set_list)

# =======================================================
mlp = MLP(inputs_size=3, hidden_layer_sizes=[4, 4])

dataset = np.array([[[1, 2, -3]],
                     [[4, -3, 1]],
                     [[-3, 2, -2]],
                     [[3, 4, -6]]])
ground_truths = np.array([1, -1, -1, 1])

result = mlp.forward_pass_all(dataset, ground_truths)
print(result)
mlp.backward_pass(result[1])
mlp.gradient_descent_optimization(0.1)
print(mlp.last_loss)
print(result)