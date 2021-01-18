# l2
l2∞normlization\\
Net on Cifar10 is the code to train a network with L_2,∞normlzation on trainingset Clfar10.

Net on MNIST is the code to train a network with L_2,∞normlzation on trainingset MNIST.

Their outputs are the accuracy on test set.

You can change four parameters.

BC:the stepsize of trainset,always 64 or 32

XLBC:the stepsize of training

Ln:the upperbound of L_2,∞normlization,you can find in paper

LS:hou many turns you want to train

recommend:

Consider of the instability of algorithm, try small XLBC and increase LS.
