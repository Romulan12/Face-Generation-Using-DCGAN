import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import conv
from layers import deconv


class Discriminator(nn.Module):

    def __init__(self, conv_dim):
        """
        Initialize the Discriminator Module
        :param conv_dim: The depth of the first convolutional layer
        """
        super(Discriminator, self).__init__()

        # complete init function
        self.conv_dim = conv_dim
        
        self.conv1 = conv(3, conv_dim, 4, batch_norm=False)
        self.conv2 = conv(conv_dim, conv_dim*2, 4, batch_norm=True)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 4, batch_norm=True)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, batch_norm=True)
        
        self.fc1 = nn.Linear(2*2*conv_dim*8, 2048)
        self.fc2 = nn.Linear(2048, 1)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: Discriminator logits; the output of the neural network
        """
        
        x = F.leaky_relu(self.conv1(x), 0.25)
        x = F.leaky_relu(self.conv2(x), 0.25)
        x = F.leaky_relu(self.conv3(x), 0.25)
        x = F.leaky_relu(self.conv4(x), 0.25)
        
        x = x.view(-1, 2*2*self.conv_dim*8)
        
        x = self.fc1(x)
        d_out = self.fc2(x)
        
        return d_out



class Generator(nn.Module):
    
    def __init__(self, z_size, conv_dim):
        """
        Initialize the Generator Module
        :param z_size: The length of the input latent vector, z
        :param conv_dim: The depth of the inputs to the *last* transpose convolutional layer
        """
        super(Generator, self).__init__()

        self.conv_dim = conv_dim
    
        self.fc1 = nn.Linear(z_size, 2048)
        self.fc2 = nn.Linear(2048, 2*2*conv_dim*8)
        
        self.t_conv1 = deconv(conv_dim*8, conv_dim*4, 4, batch_norm=True)
        self.t_conv2 = deconv(conv_dim*4, conv_dim*2, 4, batch_norm=True)
        self.t_conv3 = deconv(conv_dim*2, conv_dim, 4, batch_norm=True)
        self.t_conv4 = deconv(conv_dim, 3, 4, batch_norm=False)

    def forward(self, x):
        """
        Forward propagation of the neural network
        :param x: The input to the neural network     
        :return: A 32x32x3 Tensor image as output
        """
        x = self.fc1(x)
        x = self.fc2(x)
        
        x = x.view(-1, self.conv_dim*8, 2, 2)
        
        x = F.relu(self.t_conv1(x))
        x = F.relu(self.t_conv2(x))
        x = F.relu(self.t_conv3(x))
        x = self.t_conv4(x)
        g_out = torch.tanh(x)
        
        return g_out


