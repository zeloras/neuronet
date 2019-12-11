import numpy


class NeuronNet:
    iterations: int = 1000
    training: []
    outputs: []
    compare: []

    def __init__(self, input_array: [] = [], output_array: [] = [], to_calculate: [] = []):
        self.training = input_array
        self.outputs = output_array
        self.compare = to_calculate

    @staticmethod
    def sigmoid(i):
        return 1 / (1 + numpy.exp(-i))

    def calculate(self):
        training = numpy.array(self.training)
        outputs = numpy.array([self.outputs]).T
        weights = 2 * numpy.random.random((len(training[0]), 1)) - 1

        for i in range(self.iterations):
            outputs_data = self.sigmoid(numpy.dot(training, weights))
            error = outputs - outputs_data
            weights += numpy.dot(training.T, error * (outputs_data * (1 - outputs_data)))

        return self.sigmoid(numpy.dot(numpy.array(self.compare), weights))


out_truth_data = [1, 0, 1, 0, 0, 1]
training_data = [[0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 1]]
compare_data = [0, 0, 1, 0]

nn = NeuronNet(training_data, out_truth_data, compare_data)
print(nn.calculate())
