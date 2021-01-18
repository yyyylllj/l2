# l2
Run all files on pytorch.

Net on Cifar10.py is the code to train a network with L_2,∞normlzation on trainingset Clfar10.

Net on MNIST.py is the code to train a network with L_2,∞normlzation on trainingset MNIST.

Their outputs are the accuracy on test set.

If you want to save the parameters of network, use their line-87 and put on the path you want.

You can change four parameters in these two files.

BC:the stepsize of trainset,always 64 or 32

XLBC:the stepsize of training

Ln:the upperbound of L_2,∞normlization,you can find in paper

LS:hou many turns you want to train

For MNIST, if you want to discuss the accuracy on test set with noise, use file named "noise test.py"

You should input path when you save "noise".

You can change one parameters in this file.

L_n:the level of noise.

Then you should use the line 99-111 in file "Net on MNIST.py".

At line 99, input path which you save "noise" on.

recommend:

Consider of the instability of algorithm, try small XLBC and increase LS.



