
import torch
import torch.nn as nn
import torch.nn.functional as F


def scale(x, feature_range=(-1, 1)):
    ''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    
    min, max = feature_range
    x = x * (max -  min) + min
    return x
    
def weights_init_normal(m):
    """
    Applies initial weights to certain layers in a model .
    The weights are taken from a normal distribution 
    with mean = 0, std dev = 0.02.
    :param m: A module or layer in a network    
    """

    classname = m.__class__.__name__
    
    # TODO: Apply initial weights to convolutional and linear layers
    if (hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1)):
        nn.init.normal_(m.weight.data, mean=0, std=0.02)
   

