from neural import NeuralNet

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")

xor_tdata = [
    ([0, 0], [0]),
    ([1, 0], [1]),
    ([0, 1], [1]),
    ([0, 0], [0]),
]


xor_neuralnet = NeuralNet(2, 2, 1)

xor_neuralnet.train(xor_tdata, iters = 100, print_interval = 100)

print(xor_neuralnet.test_with_expected(xor_tdata))

print("<<<<<<<<<<<<<< XOR >>>>>>>>>>>>>>\n")







































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































voter_data = [
    ([0.9, 0.6, 0.8, 0.3, 0.1], [1.0]),
    ([0.8, 0.8, 0.4, 0.6, 0.4], [1.0]),
    ([0.7, 0.2, 0.4, 0.6, 0.3], [1.0]),
    ([0.5, 0.5, 0.8, 0.4, 0.8], [0.0]),
    ([0.3, 0.1, 0.6, 0.8, 0.8], [0.0]),
    ([0.6, 0.3, 0.4, 0.3, 0.6], [0.0]),
]


voter_nn = NeuralNet(5, 1, 1)

voter_nn.train(voter_data, iters = 10000, print_interval = 100)

print(voter_nn.test_with_expected(voter_data))

voter_test = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, 7],
    [0.8, 0.3, 0.3, 3, 0.8],
    [0.8, 0.3, 0.3, 0.8, 0.3],
    [0.9, 0.8, 0.8, 0.3, 0.6],
]
print("VOTER TEST")
voter_nn.test(voter_test)
print(voter_nn.test(voter_test))