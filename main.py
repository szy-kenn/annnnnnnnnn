from ANN import ANN
import csv
from math import floor, ceil

inputs = []
targets = []

with open("training.csv", "r") as f:
    csv_reader = csv.reader(f)
    for row in csv_reader:
        input_row = [float(elem) for elem in row[:-1]]
        inputs.append(input_row)
        targets.append(int(row[-1]))


def preprocessing(inputs):
    row = [0] * 7
    col = [0] * 5

    for idx, cell in enumerate(inputs):
        if cell == 1:
            col[idx % 5] += 1 
            row[floor(idx/5) % 7] += 1

    row.extend(col)
    return row

_neuron_count = 5
_fan_in = 2

def train():
          
        model = ANN(neuron_count=_neuron_count, 
                    fan_in=_fan_in, 
                    fan_out=1)

        weights, biases = model.train(inputs, targets,
                                learning_rate=0.001,
                                max_epoch=100000,
                                error_tolerance=1e-15)
        
        weights.reverse()
        biases.reverse()

        with open("weights.txt", "w") as f:
                f.write(str(weights))

        with open("biases.txt", "w") as f:
                f.write(str(biases))

def test():
    trained = ANN(neuron_count=_neuron_count, 
            fan_in=_fan_in, 
            fan_out=1,     
            weights=[[0.3917949289258904, 0.6715173880152004, -0.21166152414942513, 0.14835417268097964], [1.0260756002490465, 0.9851282930568723], [0.9770666139261482, 1.0130796067452523], [0.9737024307433648, 1.014998302611101], [0.9940179784191709, 1.003411728628598]],
            biases=[-2.6696000838016847e-05, -0.008654184305581115, 0.007611320465822845, 0.008727853202305836, 0.0019853624379129035]
            )

    x = trained.test([1, 5])
    print(x)

# train()
test()