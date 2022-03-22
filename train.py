import torch
import torch.optim as optim

import unit_tests.problem_unittests as tests

from DCGAN import Discriminator
from DCGAN import Generator


from utils.utilities import weights_init_normal
from dataLoader.dataLoader import get_dataloader
from utils.loss import real_loss, fake_loss

from utils.train_script import train
import matplotlib.pyplot as plt
import numpy as np

def build_network(d_conv_dim, g_conv_dim, z_size):
	
    '''  define discriminator and generator  '''
    D = Discriminator(d_conv_dim)
    G = Generator(z_size=z_size, conv_dim=g_conv_dim)

    '''  initialize model weights  '''
    D.apply(weights_init_normal)
    G.apply(weights_init_normal)

    print(D)
    print()
    print(G)
    
    return D, G

def training():

	'''  Model hyperparams  '''

	d_conv_dim = 32
	g_conv_dim = 64
	z_size = 256

	'''  Define function hyperparameters  '''
	batch_size = 64
	img_size = 32


	data_dir = 'data/processed_celeba_small'
	img_dir = data_dir

	celeba_train_loader = get_dataloader(batch_size, img_size, data_dir = img_dir)

	D, G = build_network(d_conv_dim, g_conv_dim, z_size)

	train_on_gpu = torch.cuda.is_available()
	if not train_on_gpu:
	    print('No GPU found. Please use a GPU to train your neural network.')
	else:
	    print('Training on GPU!')


	'''  Create optimizers for the discriminator D and generator G  ''' 
	d_optimizer = optim.Adam(D.parameters(), 0.002, [0.3, 0.999])
	g_optimizer = optim.Adam(G.parameters(), 0.002, [0.3, 0.999])

	# set number of epochs 
	n_epochs = 15

	train(D, G, d_optimizer, g_optimizer, real_loss, fake_loss,train_on_gpu,celeba_train_loader,  z_size = z_size, n_epochs=n_epochs , print_every=50)



'''  Run unit tests  ''' 
testOut_D = tests.test_discriminator(Discriminator)
testOut_G = tests.test_generator(Generator)
try:
	if testOut_G == 1 and testOut_D == 1:
		training()
except Exception as E:
	print(E)	 
	exit();

