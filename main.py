import numpy


def sigmoid(i):
    return 1 / (1 + numpy.exp(-i))


def calculate(training: [], outputs: [], compare: [], iterations: int = 10000):
    training = numpy.array(training)
    outputs = numpy.array([outputs]).T
    weights = 2 * numpy.random.random((len(training[0]), 1)) - 1

    for i in range(iterations):
        outputs_data = sigmoid(numpy.dot(training, weights))
        error = outputs - outputs_data
        weights += numpy.dot(training.T, error * (outputs_data * (1 - outputs_data)))

    return sigmoid(numpy.dot(numpy.array(compare), weights))


out_truth_data = [1, 0, 1, 0, 0, 1]
training_data = [[0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 0, 0, 1],
            [0, 0, 0, 0],
            [0, 1, 1, 1]]
compare_data = [0, 0, 1, 0]
print(calculate(training_data, out_truth_data, compare_data, iterations=1000))
