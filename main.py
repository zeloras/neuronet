import numpy


def sigmoid(i):
    return 1 / (1 + numpy.exp(-i))


training = numpy.array([[0, 0, 1],
                        [1, 1, 1],
                        [1, 0, 1],
                        [0, 0, 0],
                        [0, 1, 1]])

outputs = numpy.array([[0, 1, 1, 0, 0]]).T
numpy.random.seed(1)
weights = 2 * numpy.random.random((3, 1)) - 1
print("Random data")
print(weights)

for i in range(30000):
    layer = training
    outputs_data = sigmoid(numpy.dot(layer, weights))
    error = outputs - outputs_data
    adjustment = numpy.dot(layer.T, error * (outputs_data * (1 - outputs_data)))
    weights += adjustment


input_new = numpy.array([1, 1, 1])
outputs = sigmoid(numpy.dot(input_new, weights))

print(outputs)