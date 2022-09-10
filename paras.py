# setting arguments
import numpy as np
import argparse
parser = argparse.ArgumentParser(description='List default parameter set-up')
parser.add_argument('--Index', type=int, default=1, help='dataset index')
parser.add_argument('--noise', type=int, default=10, help='magnitude of noise')

# NN superparameters
parser.add_argument('--width', type=int, default=64, help='width of the latent layer')
parser.add_argument('--depth', type=int, default=3, help='depth of the latent layer')

# Optimizators
parser.add_argument('--lr', type=np.float32, default=1e-3, help='initial learning rate')
parser.add_argument('--l2', type=np.float32, default=1e-6, help='l2 regularization')
parser.add_argument('--slambda', type=np.float32, default=1e-3, help='smoothness condition')
parser.add_argument('--epochs', type=int, default=1600, help='Number of epochs to warm up.')
parser.add_argument('--maxiter', type=int, default=10000, help='Max epochs in training.')


args = parser.parse_args()
print('initial lr = ',args.lr)
print('initial slambda = ',args.slambda)
print('initial l2 = ',args.l2)
print('magnitude of noise = ',args.noise)
