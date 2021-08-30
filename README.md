# l2
Run all files on pytorch. Please configure the environment with in the Requirement.txt.

Net on MNIST.py is the code to train a network with L_2,∞normlzation on trainingset MNIST. The program can run directly.

You can change four parameters in these the file.

BC:the stepsize of trainset,always 64 or 32

XLBC:the stepsize of training

Ln:the upperbound of L_2,∞normlization,you can find in paper

LS:hou many turns you want to train

(We have show a set of appropriate parameters in the files, which can be used directly.)

Its outputs are the accuracy of each round on test set in the training.

If you want to save the parameters of network, line-89 of "Net on MNIST.py".

If you do not want to train, 'mod_MNIST.pt' is a network which had been trained in 'Net on MNIST.py'. And you can use 'Net of MNIST has been trained.py' to test its accuracy.

If you want to discuss the accuracy on test set with noise, use file named "noise test.py"

Maybe you need to input path when you save "noise" in the line 17.

You can change one parameters in this file.

L_n:the level of noise.

Then you should use the line 90-102 in file "Net on MNIST.py".

At line 90, may be you nedd to input path which you save "noise" on.





