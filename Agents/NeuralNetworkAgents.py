from models import FullyConnected #, GraphNeuralNetwork, EdgeGNN
from .ReplayMemory import ReplayMemory #, GNNReplayMemory
import torch
import torch.optim as optim

''' 
This class contains the policy and target neural network for a certain destination node deep Q-learning.
node_number: this is the ID associated with each network, since each neural network is associated with some destination node 
network_size: this is the number of nodes in our network, gives us part of the size of our hidden layer
num_extra: the number of extra parameters we will be supplying in our network (network_size + num_extra = hidden layer size)
capacity: the maximum memories that our neural network can sample from 
'''
class DQN(object):
    def __init__(self, node_number, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        self.ID = node_number
        # action_space = setting["IAB_Config"]["number Parents"] + setting["IAB_Config"]["number Childrens"] + setting["IAB_Config"]["Max Users"] if setting["NETWORK"]["number user"] != 0 else setting["NETWORK"]["number Basestation"]
        action_space = setting["NETWORK"]["number user"] + setting["NETWORK"]["number Basestation"] #setting["IAB_Config"]["number Parents"] + setting["IAB_Config"]["number Childrens"] + setting["IAB_Config"]["Max Users"] if setting["NETWORK"]["number user"] != 0 else setting["NETWORK"]["number Basestation"]
        self.policy_net = FullyConnected.FullyConnectedNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net = FullyConnected.FullyConnectedNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_memory = ReplayMemory(setting['DQN']['memory_bank_size'])
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

class ValueDeepNetwork(object):
    def __init__(self, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        self.network = FullyConnected.FullyConnectedNetwork(state_dim, action_space=1).to(self.device)
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate)

class RecurrentValueDeepNetwork(object):
    def __init__(self, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        self.network = FullyConnected.RecurrentNetwork(state_dim, action_space=1).to(self.device)
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate)

class RelationalRecurrentDQN(object):
    def __init__(self, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        action_space = setting['NETWORK']['number Basestation']
        self.policy_net = FullyConnected.RecurrentNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net = FullyConnected.RecurrentNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_memory = ReplayMemory(setting['DQN']['memory_bank_size'])
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

class LongLengthRecurrentValueDeepNetwork(object):
    def __init__(self, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        self.network = FullyConnected.LongLengthRecurrentNetwork(state_dim, action_space=1).to(self.device)
        self.optimizer = optim.Adam(params=self.network.parameters(), lr=learning_rate)



class LongLengthRelationalRecurrentDQN(object):
    def __init__(self, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        action_space = setting['NETWORK']['number Basestation']
        self.policy_net = FullyConnected.LongLengthRecurrentNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net = FullyConnected.LongLengthRecurrentNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_memory = ReplayMemory(setting['DQN']['memory_bank_size'])
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

class RelationalDQN(object):
    def __init__(self, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        action_space = setting['NETWORK']['number Basestation']
        self.policy_net = FullyConnected.FullyConnectedNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net = FullyConnected.FullyConnectedNetwork(state_dim, action_space=action_space).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_memory = ReplayMemory(setting['DQN']['memory_bank_size'])
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

class DQNWithEmbedding:
    def __init__(self, node_number, state_dim, setting, device):
        self.device = device
        learning_rate = setting['DQN']['optimizer_learning_rate']
        self.ID = node_number
        self.policy_net = FullyConnected.FullyConnectedEmbeddingNetwork(state_dim, num_nodes=setting["NETWORK"]["number Basestation"]).to(self.device)
        self.target_net = FullyConnected.FullyConnectedEmbeddingNetwork(state_dim, num_nodes=setting["NETWORK"]["number Basestation"]).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.replay_memory = ReplayMemory(setting['DQN']['memory_bank_size'])
        self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)

# class GDQN(object):
#     def __init__(self, network_size, state_space, setting, device):
#         self.device = device
#         learning_rate = setting['DQN']['optimizer_learning_rate']
#         message_passing_config = {'num_layers': 1, 'node_state_dim': 128, 'in_edge_channels': 1,
#                               'hidden_edge_channels': [16, 4, 1], 'hidden_edge_channels_conv': 16, 'node_embedding_dim': 128}
#         node_embedding_config = {'input_dim': state_space, 'fc_dims': [128]}
#         self.policy_net = EdgeGNN.RouteFinderGNN(message_passing=message_passing_config, learned_NodeEncoder_dic=node_embedding_config).to(self.device)
#         self.target_net = EdgeGNN.RouteFinderGNN(message_passing=message_passing_config, learned_NodeEncoder_dic=node_embedding_config).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
#         self.replay_memory = GNNReplayMemory(setting['DQN']['memory_bank_size'])
#         self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
#
# class simpleGraphDQN(object):
#     def __init__(self, network_size, state_space, setting, device):
#         self.device = device
#         learning_rate = setting['DQN']['optimizer_learning_rate']
#         self.policy_net = EdgeGNN.simpleRouteFinderGNN(state_dim=state_space, action_dim=110).to(self.device)
#         self.target_net = EdgeGNN.simpleRouteFinderGNN(state_dim=state_space, action_dim=110).to(self.device)
#         self.target_net.load_state_dict(self.policy_net.state_dict())
#         self.target_net.eval()
#         self.replay_memory = GNNReplayMemory(setting['DQN']['memory_bank_size'])
#         self.optimizer = optim.Adam(params=self.policy_net.parameters(), lr=learning_rate)
#
