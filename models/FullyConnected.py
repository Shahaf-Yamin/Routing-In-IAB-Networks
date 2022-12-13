import torch.nn as nn
'''Class created to configure the structure for our neural networks'''
import torch
import numpy as np

class FullyConnectedNetwork(nn.Module):
    '''
    Initialize a neural network for deep Q-learning.
    num_states: the number of nodes in the network
    and also the number of outputs for the network.
    num_extra_params: the number of other parameters besides for the
    one-hot encoded vector for the current node we are taking in.
    Our neural network has 3 layers and 1 hidden layer.
    '''

    def __init__(self, state_dim, action_space):
        super(FullyConnectedNetwork, self).__init__()
        hidden_size = 128 # min(2 * state_dim, 128)
        """ Inputs to hidden layer linear transformation """
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        """ HL1 to output linear transformation """
        self.layer3 = nn.Linear(hidden_size, action_space)
        """ our activation function  """
        self.activation = nn.LeakyReLU()

    ''' Output the result of the network given input x.
    Take in input x, transform to hidden layer, apply tanh to hidden layer,
    and transform them into outputs. '''

    def forward(self, x):
        """  HL1 with tanh activation """
        out = self.activation(self.layer1(x))
        out = self.activation(self.layer2(out))
        """  Output layer with linear activation """
        out = self.layer3(out)
        return out

class RecurrentNetwork(nn.Module):
    '''
    Initialize a neural network for deep Q-learning.
    num_states: the number of nodes in the network
    and also the number of outputs for the network.
    num_extra_params: the number of other parameters besides for the
    one-hot encoded vector for the current node we are taking in.
    Our neural network has 3 layers and 1 hidden layer.
    '''

    def __init__(self, state_dim, action_space):
        super(RecurrentNetwork, self).__init__()
        hidden_size = 128 # min(2 * state_dim, 128)
        """ Inputs to hidden layer linear transformation """
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.rec_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0)
        self.hidden_state = np.random.randn(1, 1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        """ HL1 to output linear transformation """
        self.layer3 = nn.Linear(hidden_size, action_space)
        """ our activation function  """
        self.activation = nn.LeakyReLU()

    ''' Output the result of the network given input x.
    Take in input x, transform to hidden layer, apply tanh to hidden layer,
    and transform them into outputs. '''

    def forward(self, x):
        """  HL1 with tanh activation """
        out = self.activation(self.layer1(x))
        out = torch.unsqueeze(out, dim=0)
        output, hn = self.rec_layer(out, torch.tensor(self.hidden_state, dtype=torch.float32))
        out = torch.squeeze(output, dim=0)
        self.hidden_state = hn.detach().numpy()
        out = self.activation(self.layer2(out))
        """  Output layer with linear activation """
        out = self.layer3(out)
        return out

class LongLengthRecurrentNetwork(nn.Module):
    '''
    Initialize a neural network for deep Q-learning.
    num_states: the number of nodes in the network
    and also the number of outputs for the network.
    num_extra_params: the number of other parameters besides for the
    one-hot encoded vector for the current node we are taking in.
    Our neural network has 3 layers and 1 hidden layer.
    '''

    def __init__(self, state_dim, action_space):
        super(LongLengthRecurrentNetwork, self).__init__()
        hidden_size = 128 # min(2 * state_dim, 128)
        """ Inputs to hidden layer linear transformation """
        self.layer1 = nn.Linear(state_dim, hidden_size)
        self.rec_layer = nn.GRU(input_size=hidden_size, hidden_size=hidden_size, batch_first=True, dropout=0)
        self.hidden_state = np.random.randn(1, 1, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        """ HL1 to output linear transformation """
        self.layer3 = nn.Linear(hidden_size, action_space)
        """ our activation function  """
        self.activation = nn.LeakyReLU()

    ''' Output the result of the network given input x.
    Take in input x, transform to hidden layer, apply tanh to hidden layer,
    and transform them into outputs. '''

    def forward(self, xx):
        """  HL1 with tanh activation """
        outputs = []
        hn = torch.tensor(self.hidden_state, dtype=torch.float32)
        for x in xx:
            out = self.activation(self.layer1(x))
            out = torch.unsqueeze(out, dim=0)
            output, hn = self.rec_layer(out, hn)
            out = torch.squeeze(output, dim=0)
            out = self.activation(self.layer2(out))
            """  Output layer with linear activation """
            out = self.layer3(out)
            outputs.append(out.squeeze())
        outputs = torch.stack(outputs)
        if len(xx) == 1:
            ''' Update the hidden state '''
            self.hidden_state = hn.detach().numpy()
        return outputs


class FullyConnectedEmbeddingNetwork(nn.Module):
    '''
    Initialize a neural network for deep Q-learning.
    num_states: the number of nodes in the network
    and also the number of outputs for the network.
    num_extra_params: the number of other parameters besides for the
    one-hot encoded vector for the current node we are taking in.
    Our neural network has 3 layers and 1 hidden layer.
    '''

    def __init__(self, state_dim, num_nodes):
        super(FullyConnectedEmbeddingNetwork, self).__init__()
        hidden_size = 16
        """ Inputs to Embedding layers """
        self.input_layers = [nn.Linear(state, hidden_size, bias=False) for state in state_dim]
        """ HL1 to output linear transformation """
        self.layer1 = nn.Linear(hidden_size * len(state_dim), hidden_size)
        """ HL2 to output linear transformation """
        self.layer2 = nn.Linear(hidden_size, num_nodes)
        """ our activation function  """
        self.activation = nn.ReLU()

    ''' Output the result of the network given input x.
    Take in input x, transform to hidden layer, apply activation function to the hidden layer,
    and transform them into outputs. '''

    def forward(self, inputs):
        output = []
        for x, layer in zip(inputs, self.input_layers):
            output.append(self.activation(layer(x)))
        """  Output layer with linear activation """
        x = torch.cat(tuple(output), 1)
        x = self.activation(self.layer1(x))
        out = self.layer2(x)
        return out
