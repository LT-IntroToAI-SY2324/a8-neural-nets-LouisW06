from neural import NeuralNet

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_tdata = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([0, 0], [0]),
]


xor_neuralnet = NeuralNet(2, 2, 1)

xor_neuralnet.train(xor_tdata, iters = 1000, print_interval = 1000)

print(xor_neuralnet.test_with_expected(xor_tdata))

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")
