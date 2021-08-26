# l2
Run all files on pytorch. Please configure the environment with in the Requirement.txt.

Net on Cifar10.py is the code to train a network with L_2,∞normlzation on trainingset Clfar10.

Net on MNIST.py is the code to train a network with L_2,∞normlzation on trainingset MNIST.

The program can run directly. Their outputs are the accuracy of each round on test set in the training.

If you want to save the parameters of network, use line-87 of "Net on Cifar10.py" and line-77 of "Net on MNIST.py" and put on the path you want.

You can change four parameters in these two files.

BC:the stepsize of trainset,always 64 or 32

XLBC:the stepsize of training

Ln:the upperbound of L_2,∞normlization,you can find in paper

LS:hou many turns you want to train

(We have show a set of appropriate parameters in the each files.)

For MNIST, if you want to discuss the accuracy on test set with noise, use file named "noise test.py"

You should input path when you save "noise".

You can change one parameters in this file.

L_n:the level of noise.

Then you should use the line 89-101 in file "Net on MNIST.py".

At line 89, input path which you save "noise" on.





