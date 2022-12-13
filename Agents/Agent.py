import numpy as np
import random
import pandas as pd
from collections import namedtuple
import torch
from .NeuralNetworkAgents import DQN, RelationalDQN,ValueDeepNetwork, RecurrentValueDeepNetwork, RelationalRecurrentDQN, LongLengthRecurrentValueDeepNetwork, LongLengthRelationalRecurrentDQN
import torch.nn.functional as F
from torch.distributions import Categorical
import networkx as nx
from operator import itemgetter
import copy

'''
    The agent file defines a learning agent for different agents
    Currently Supported Agents:
    1. Q-Routing Agent
    2. Full-Echo Q-Routing Agent
    2. Backpressure Agent 
    3. Centralized DQN Agent (Q-Routing Based)
    4. Decentralized DQN Agent (Q-Routing Based)
'''

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward'))

class QAgent(object):
    '''
    Class contains functions:
    generate_q_table: initialize Q-table
    act: returns which next node to send packet to
    learn: update Q-table after receiving corresponding rewards
    update_epsilon: update the exploration rate
    save_agent: Saves the agent's table to a pickle file for future usage
    load_agent: Load the agent's table from a pickle file for restoring a trained agent's capabilities
    '''
    def __init__(self, dynetwork, setting, state_dim, device):
        """ 
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy 
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values 
        
        """
        self.config = {
            "learning_rate": setting['AGENT']['learning_rate'],
            "epsilon": setting['AGENT']['epsilon'],
            "epsilon_min": setting['AGENT']['epsilon_min'],
            "discount": setting['AGENT']['gamma_for_next_q_val'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "update_epsilon": False,
            }
        self.q = self.generate_q_table(dynetwork)
        self.numDest = dynetwork.available_destinations
        self.numBs = dynetwork.numBS

    ''' Use this function to set up the q-table'''
    def generate_q_table(self, dynetwork):
        q_table = {}
        available_destinations = dynetwork.available_destinations
        self.available_sources = np.arange(dynetwork.numBS)
        for currpos in self.available_sources:
            nlist = list(dynetwork._network.neighbors(currpos))
            for dest in available_destinations:
                q_table[(currpos, dest)] = {}
                for action in nlist:
                    # if dest == action or action in self.available_sources:
                    #     ''' Initialize 0 Q-table except destination '''
                        # q_table[(currpos, dest)][action] = -1000
                    ''' Initialize using Shortest Path '''
                    try:
                        q_table[(currpos, dest)][action] = nx.shortest_path_length(dynetwork._network, action, dest)
                    except nx.NetworkXNoPath:
                        q_table[(currpos, dest)][action] = -np.infty
                q_table[(currpos, dest)][dest] = 0

        for dest in available_destinations:
            q_table[(dest, dest)] = {None: 0}
        return q_table

    '''Returns action for a given state by following the greedy policy. '''
    def act(self, state, neighbor):
        ''' We will either random walk or reference Q-table with probability epsilon '''
        if random.uniform(0, 1) < max(self.config['epsilon'], self.config["epsilon_min"]):
            """ checks if the packet's current node has any available neighbors """
            available_guess = [n for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])]
            if not bool(available_guess):
                return None
            else:
                next_step = random.choice(available_guess)  # Explore action space
        else:
            temp_neighbor_dict = {n: self.q[state][n] for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])}
            """ checks if the packet's current node has any available neighbors """
            if not bool(temp_neighbor_dict):
                return None
            else:
                next_step = max(temp_neighbor_dict, key=temp_neighbor_dict.get)
        return next_step

    ''' update the exploration rate in a decaying manner'''
    def update_epsilon(self):
        self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """
    def learn(self, current_event, action, reward, done, neighbors, next_neighbors_list):
        if (action == None) or (reward == None):
            TD_error = None
        else:
            n, dest = current_event
            # Calculate dynamically the next state action space
            available_guess = [n for n in self.q[(action, dest)] if n in next_neighbors_list and np.isfinite(self.q[(action, dest)][n])]
            # Follow the greedy policy w.r.t the action value estimation from the next state action space
            max_q = 0 if done else max((self.q[(action, dest)][key] for key in available_guess))

            """ Q learning algorithm """
            TD_error = (reward + self.config["discount"] * max_q - self.q[(n, dest)][action])
            self.q[(n, dest)][action] = self.q[(n, dest)][action] + (self.config["learning_rate"])*TD_error
        informationExchangeSize = 32
        return TD_error, informationExchangeSize

    ''' Saves the agent's table to a pickle file for future usage '''
    def save_agent(self, path):
        pd.to_pickle(self.q, path + "data/agent.pkl")

    ''' Load the agent's table from a pickle file for restoring a trained agent's capabilities '''
    def load_agent(self, path):
        self.q = pd.read_pickle(path + "data/agent.pkl")

class QAgentWithTTL(QAgent):
    '''
    Class contains functions:
    generate_q_table: initialize Q-table
    act: returns which next node to send packet to
    learn: update Q-table after receiving corresponding rewards
    update_epsilon: update the exploration rate
    save_agent: Saves the agent's table to a pickle file for future usage
    load_agent: Load the agent's table from a pickle file for restoring a trained agent's capabilities
    '''

    def __init__(self, dynetwork, setting, state_dim, device):
        """
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values

        """
        self.TTL = setting["NETWORK"]["MAX_TTL"] + 1
        self.numTTLLevels = setting["NETWORK"]["QUANTIZELevels"]
        self.quantizeLevels = np.linspace(0, self.TTL, self.numTTLLevels+1)
        super().__init__(dynetwork, setting, state_dim, device)
    def quantized_ttl(self, x):
        return max(np.argwhere(self.quantizeLevels>=x)[0, 0] - 1, 0)

    def act(self, state, neighbor):
        src, dest, arrivalTime = state
        state = (src,dest,self.quantized_ttl(arrivalTime))
        ''' We will either random walk or reference Q-table with probability epsilon '''
        if random.uniform(0, 1) < max(self.config['epsilon'], self.config["epsilon_min"]):
            """ checks if the packet's current node has any available neighbors """
            available_guess = [n for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])]
            if not bool(available_guess):
                return None
            else:
                next_step = random.choice(available_guess)  # Explore action space
        else:
            temp_neighbor_dict = {n: self.q[state][n] for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])}
            """ checks if the packet's current node has any available neighbors """
            if not bool(temp_neighbor_dict):
                return None
            else:
                next_step = max(temp_neighbor_dict, key=temp_neighbor_dict.get)
        return next_step


    ''' Use this function to set up the q-table'''

    def generate_q_table(self, dynetwork):
        q_table = {}
        available_destinations = dynetwork.available_destinations
        self.available_sources = np.arange(dynetwork.numBS)
        for currpos in self.available_sources:
            nlist = list(dynetwork._network.neighbors(currpos))
            for dest in available_destinations:
                for time_idx in range(self.quantized_ttl(self.TTL)+1):
                    q_table[(currpos, dest, time_idx)] = {}
                for action in nlist:
                    for time_idx in range(self.quantized_ttl(self.TTL)+1):
                        # if dest == action or action in self.available_sources:
                        #     ''' Initialize 0 Q-table except destination '''
                        # q_table[(currpos, dest)][action] = -1000
                        ''' Initialize using Shortest Path '''
                        try:
                            q_table[(currpos, dest, time_idx)][action] = nx.shortest_path_length(dynetwork._network, action, dest)
                        except nx.NetworkXNoPath:
                            q_table[(currpos, dest, time_idx)][action] = -np.infty
                for time_idx in range(self.quantized_ttl(self.TTL)+1):
                    q_table[(currpos, dest, time_idx)][dest] = 0

        for dest in available_destinations:
            for time_idx in range(self.quantized_ttl(self.TTL)+1):
                q_table[(dest, dest, time_idx)] = {None: 0}
        return q_table

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """

    def learn(self, current_event, action, reward, done, neighbors, next_neighbors_list):
        if (action == None) or (reward == None):
            TD_error = None
        else:
            n, dest, arrival_time = current_event
            next_node_time = self.quantized_ttl(arrival_time - reward)
            arrival_time = self.quantized_ttl(arrival_time)
            # Calculate dynamically the next state action space
            available_guess = [n for n in self.q[(action, dest, next_node_time)] if
                               n in next_neighbors_list and np.isfinite(self.q[(action, dest, next_node_time)][n])]
            # Follow the greedy policy w.r.t the action value estimation from the next state action space
            max_q = 0 if done else max(self.q[(action, dest, next_node_time)][key] for key in available_guess)

            """ Q learning algorithm """
            TD_error = (reward + self.config["discount"] * max_q - self.q[(n, dest, arrival_time)][action])
            self.q[(n, dest, arrival_time)][action] = self.q[(n, dest, arrival_time)][action] + (self.config["learning_rate"]) * TD_error
        informationExchangeSize = 32
        return TD_error, informationExchangeSize

class QAgentWithKnownUserAssociation(QAgent):
    '''
    Class contains functions:
    generate_q_table: initialize Q-table
    act: returns which next node to send packet to
    learn: update Q-table after receiving corresponding rewards
    update_epsilon: update the exploration rate
    save_agent: Saves the agent's table to a pickle file for future usage
    load_agent: Load the agent's table from a pickle file for restoring a trained agent's capabilities
    '''

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)

    ''' Use this function to set up the q-table'''

    def generate_q_table(self, dynetwork):
        q_table = {}
        available_destinations = np.arange(dynetwork.numBS)
        self.available_sources = np.arange(dynetwork.numBS)
        for currpos in self.available_sources:
            nlist = list(dynetwork._network.neighbors(currpos))
            for dest in available_destinations:
                q_table[(currpos, dest)] = {}
                for action in nlist:
                    # if dest == action or action in self.available_sources:
                    #     ''' Initialize 0 Q-table except destination '''
                    # q_table[(currpos, dest)][action] = -1000
                    ''' Initialize using Shortest Path '''
                    try:
                        q_table[(currpos, dest)][action] = nx.shortest_path_length(dynetwork._network, action, dest)
                    except nx.NetworkXNoPath:
                        q_table[(currpos, dest)][action] = -np.infty
                q_table[(currpos, dest)][dest] = 0

        for dest in available_destinations:
            q_table[(dest, dest)] = {None: 0}
        return q_table

    '''Returns action for a given state by following the greedy policy. '''

    def act(self, state, neighbor, dest):
        ''' We will either random walk or reference Q-table with probability epsilon '''
        src, dest, associationUser = state

        if dest in neighbor:
            return (dest, (src, dest))
        states = [(src, idx) for idx in range(associationUser.shape[0]) if associationUser[idx] == 1]
        if random.uniform(0, 1) < max(self.config['epsilon'], self.config["epsilon_min"]):
            """ checks if the packet's current node has any available neighbors """
            available_guess = set()
            for state in states:
                available_guess = available_guess.union(set(n for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])))
            if not bool(available_guess):
                return None
            else:
                # Explore action space
                next_step = random.choice(list(available_guess))
                chosenDest = random.choice(list(map(lambda x: x[1], states)))
        else:
            temp_neighbor_dict = {}
            for state in states:
                temp_neighbor_dict[state] = {n: self.q[state][n] for n in self.q[state] if n in neighbor}
            """ checks if the packet's current node has any available neighbors """
            if not bool(temp_neighbor_dict):
                return None
            else:
                values = {state:-np.infty for state in states}
                for state in states:
                    next_step = max(temp_neighbor_dict[state], key=temp_neighbor_dict[state].get)
                    values[state] = (next_step, temp_neighbor_dict[state][next_step])
                bestAction, bestValue = values[states[0]]
                chosenDest = states[0][1]
                for state in states:
                    curVal = values[state][1]
                    if curVal > bestValue:
                        chosenDest = state[1]
                        bestValue = curVal
                        bestAction = values[state][0]
                next_step = bestAction
        return next_step, (src, chosenDest)

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """
    def learn(self, current_event, action, reward, done, neighbors, next_neighbors_list):
        if (action == None) or (reward == None):
            TD_error = None
        else:
            src, dest, associationUser = current_event
            partial_action = action[0]
            # Calculate dynamically the next state action space
            if dest in next_neighbors_list:
                done = True
                max_q = 0
            else:
                dests = np.argwhere(associationUser == 1).squeeze().numpy()
                available_value = [0]
                for bs_dest in dests:
                    available_value = [self.q[(partial_action, bs_dest)][n] for n in self.q[(partial_action, bs_dest)] if n in next_neighbors_list]
                # Follow the greedy policy w.r.t the action value estimation from the next state action space
                max_q = 0 if done else max(available_value)
            chosenState = action[1]

            """ Q learning algorithm """
            TD_error = (reward + self.config["discount"] * max_q - self.q[chosenState][partial_action])
            self.q[chosenState][partial_action] = self.q[chosenState][partial_action] + (self.config["learning_rate"]) * TD_error
        informationExchangeSize = 32
        return TD_error, informationExchangeSize

class QAgentWithTTLWithKnownUserAssociation(QAgentWithTTL):
    '''
    Class contains functions:
    generate_q_table: initialize Q-table
    act: returns which next node to send packet to
    learn: update Q-table after receiving corresponding rewards
    update_epsilon: update the exploration rate
    save_agent: Saves the agent's table to a pickle file for future usage
    load_agent: Load the agent's table from a pickle file for restoring a trained agent's capabilities
    '''

    def __init__(self, dynetwork, setting, state_dim, device):
        """
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values

        """
        super().__init__(dynetwork, setting, state_dim, device)


    def generate_q_table(self, dynetwork):
        q_table = {}
        available_destinations = np.arange(dynetwork.numBS)
        self.available_sources = np.arange(dynetwork.numBS)
        for currpos in self.available_sources:
            nlist = list(dynetwork._network.neighbors(currpos))
            for dest in available_destinations:
                for time_idx in range(self.TTL):
                    q_table[(currpos, dest,time_idx)] = {}
                for action in nlist:
                    for time_idx in range(self.quantized_ttl(self.TTL)+1):
                        # if dest == action or action in self.available_sources:
                        #     ''' Initialize 0 Q-table except destination '''
                        # q_table[(currpos, dest)][action] = -1000
                        ''' Initialize using Shortest Path '''
                        try:
                            q_table[(currpos, dest, time_idx)][action] = nx.shortest_path_length(dynetwork._network, action, dest)
                        except nx.NetworkXNoPath:
                            q_table[(currpos, dest, time_idx)][action] = -np.infty
                for time_idx in range(self.quantized_ttl(self.TTL)+1):
                    q_table[(currpos, dest, time_idx)][dest] = 0

        for dest in available_destinations:
            for time_idx in range(self.quantized_ttl(self.TTL)+1):
                q_table[(dest, dest, time_idx)] = {None: 0}
        return q_table

    ''' Use this function to set up the q-table'''
    def act(self, state, neighbor, dest):
        ''' We will either random walk or reference Q-table with probability epsilon '''
        src, dest, arrivalTime, associationUser = state

        if dest in neighbor:
            return (dest, (src, dest, self.quantized_ttl(arrivalTime)))
        states = [(src, idx, self.quantized_ttl(arrivalTime)) for idx in range(associationUser.shape[0]) if associationUser[idx] == 1]
        if random.uniform(0, 1) < max(self.config['epsilon'], self.config["epsilon_min"]):
            """ checks if the packet's current node has any available neighbors """
            available_guess = set()
            for state in states:
                available_guess = available_guess.union(set(n for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])))
            if not bool(available_guess):
                return None
            else:
                # Explore action space
                next_step = random.choice(list(available_guess))
                chosenDest = random.choice(list(map(lambda x: x[1], states)))
        else:
            temp_neighbor_dict = {}
            for state in states:
                temp_neighbor_dict[state] = {n: self.q[state][n] for n in self.q[state] if n in neighbor}
            """ checks if the packet's current node has any available neighbors """
            if not bool(temp_neighbor_dict):
                return None
            else:
                values = {state: -np.infty for state in states}
                for state in states:
                    next_step = max(temp_neighbor_dict[state], key=temp_neighbor_dict[state].get)
                    values[state] = (next_step, temp_neighbor_dict[state][next_step])
                bestAction, bestValue = values[states[0]]
                chosenDest = states[0][1]
                for state in states:
                    curVal = values[state][1]
                    if curVal > bestValue:
                        chosenDest = state[1]
                        bestValue = curVal
                        bestAction = values[state][0]
                next_step = bestAction
        return next_step, (src, chosenDest, self.quantized_ttl(arrivalTime))

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """

    def learn(self, current_event, action, reward, done, neighbors, next_neighbors_list):
        if (action == None) or (reward == None):
            TD_error = None
        else:
            src, dest, arrival_time, associationUser = current_event
            partial_action = action[0]
            next_node_time = self.quantized_ttl(arrival_time - reward)
            # Calculate dynamically the next state action space
            if dest in next_neighbors_list:
                max_q = 0
            else:
                dests = np.argwhere(associationUser == 1).squeeze().numpy()
                available_value = [0]
                for bs_dest in dests:
                    available_value = [self.q[(partial_action, bs_dest, next_node_time)][n] for n in self.q[(partial_action, bs_dest, next_node_time)] if n in next_neighbors_list]
                # Follow the greedy policy w.r.t the action value estimation from the next state action space
                max_q = 0 if done else max(available_value)
            chosenState = action[1]

            """ Q learning algorithm """
            TD_error = (reward + self.config["discount"] * max_q - self.q[chosenState][partial_action])
            self.q[chosenState][partial_action] = self.q[chosenState][partial_action] + (self.config["learning_rate"]) * TD_error
        informationExchangeSize = 32
        return TD_error, informationExchangeSize

class FullEchoQAgent(QAgent):
    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.config['epsilon'] = 0.0
        self.config['epsilon_min'] = 0.0

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """

    def learn(self, current_event, action, rewards, dones, neighbors, next_state_neighbors):
        if (action == None) or (rewards == None):
            TD_error = None
        else:
            state, neighbor_list = current_event
            n, dest = state
            TD_errors = np.zeros((len(neighbor_list), 1))

            ''' Iterate through all available neighbors '''
            for cnt, info in enumerate(zip(neighbor_list, rewards, dones, next_state_neighbors)):
                neighbor, reward, done, neighbor_available_neighbors = info
                # Calculate dynamically the next state action space
                available_guess = [n for n in self.q[(neighbor, dest)] if n in neighbor_available_neighbors and np.isfinite(self.q[(neighbor, dest)][n])]
                # Follow the greedy policy w.r.t the action value estimation from the next state action space
                max_q = 0 if done else max((self.q[(neighbor, dest)][key] for key in available_guess))
                """ Q learning algorithm """
                TD_error = (reward + self.config["discount"] * max_q * (1 - done) - self.q[(n, dest)][neighbor])
                self.q[(n, dest)][neighbor] = self.q[(n, dest)][neighbor] + (self.config["learning_rate"]) * TD_error
                TD_errors[cnt] = TD_error
            TD_error = np.mean(TD_errors)
        informationExchangeSize = 32 * len(neighbor_list)
        return TD_error, informationExchangeSize

class MultipleEchoQAgent(QAgent):
    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.config['epsilon'] = 0.0
        self.config['epsilon_min'] = 0.0
        self.destination_frequency = setting['AGENT']['destination_sample_size']

    """ updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """
    def learn(self, current_event, action, rewards, dones, next_state_neighbors):
        if (action == None) or (rewards == None):
            TD_error = None
        else:
            state, neighbor_list = current_event
            n, dest = state
            TD_errors = np.zeros((len(neighbor_list), 1))

            ''' Iterate through all available neighbors '''
            for cnt, info in enumerate(zip(neighbor_list, rewards, dones, next_state_neighbors)):
                neighbor, reward, done, neighbor_available_neighbors = info
                # Calculate dynamically the next state action space
                available_guess = [n for n in self.q[(neighbor, dest)] if n in neighbor_available_neighbors and np.isfinite(self.q[(neighbor, dest)][n])]
                # Follow the greedy policy w.r.t the action value estimation from the next state action space
                max_q = 0 if done else max((self.q[(neighbor, dest)][key] for key in available_guess))
                """ Q learning algorithm """
                TD_error = (reward + self.config["discount"] * max_q * (1 - done) - self.q[(n, dest)][neighbor])
                self.q[(n, dest)][neighbor] = self.q[(n, dest)][neighbor] + (self.config["learning_rate"]) * TD_error
                TD_errors[cnt] = TD_error

            for cnt, info in enumerate(zip(neighbor_list, rewards, dones, next_state_neighbors)):
                neighbor, reward, done, neighbor_available_neighbors = info
                if neighbor < self.numBs:
                    # this case we sample further information regarding another destinations.
                    for sample in range(self.destination_frequency):
                        # sample uniformly a random destination
                        dest = np.random.choice(self.numDest)
                        # Calculate dynamically the next state action space
                        available_guess = [n for n in self.q[(neighbor, dest)] if n in neighbor_available_neighbors and np.isfinite(self.q[(neighbor, dest)][n])]
                        if dest in available_guess: max_q = self.q[(neighbor, dest)][dest]
                        # Follow the greedy policy w.r.t the action value estimation from the next state action space
                        else: max_q = max((self.q[(neighbor, dest)][key] for key in available_guess))
                        """ Q learning algorithm """
                        TD_error = (reward + self.config["discount"] * max_q * (1 - done) - self.q[(n, dest)][neighbor])
                        self.q[(n, dest)][neighbor] = self.q[(n, dest)][neighbor] + (self.config["learning_rate"]) * TD_error
                else:
                    for sample in range(self.destination_frequency):
                        # sample uniformly a random destination
                        dest = np.random.choice(self.numDest)
                        TD_error = (reward - self.q[(n, dest)][dest])
                        self.q[(n, dest)][dest] = self.q[(n, dest)][dest] + (self.config["learning_rate"]) * TD_error
            TD_error = np.mean(TD_errors)
            informationExchangeSize = 32 * len(neighbor_list) * (1 + self.destination_frequency)
        return TD_error, informationExchangeSize

class CQAgent(QAgent):
    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.confidence_table = self.generate_confidence_table(dynetwork=dynetwork)
        self.decay = 0.9999
        self.eta = 0

    def generate_confidence_table(self, dynetwork):
        confidence_table = {}
        available_destinations = dynetwork.available_destinations
        self.available_sources = np.arange(dynetwork.numBS)
        for currpos in self.available_sources:
            nlist = list(dynetwork._network.neighbors(currpos))
            for dest in available_destinations:
                confidence_table[(currpos, dest)] = {}
                for action in nlist:
                    ''' Initialize using Shortest Path '''
                    try:
                        path = nx.shortest_path_length(dynetwork._network, action, dest)
                        confidence_table[(currpos, dest)][action] = 1
                    except nx.NetworkXNoPath:
                        confidence_table[(currpos, dest)][action] = 0
        for dest in available_destinations:
            confidence_table[(dest, dest)] = {None: 1}

        return confidence_table

    """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """

    def learn(self, current_event, action, reward, done):
        if (action == None) or (reward == None):
            TD_error = None
        else:
            n, dest = current_event
            max_q_idx = np.argmax(self.q[(action, dest)].values())
            max_next_idx = list(self.q[(action, dest)].keys())[max_q_idx]
            next_max_q = self.q[(action, dest)][max_next_idx]
            next_conf = self.confidence_table[(action, dest)][max_next_idx]
            old_conf = self.confidence_table[(n, dest)][action]
            self.eta = max(next_conf, 1 - old_conf)
            """ Q learning algorithm """
            TD_error = (reward + self.config["discount"] * next_max_q * (1 - done) - self.q[(n, dest)][action])
            self.q[(n, dest)][action] = self.q[(n, dest)][action] + self.eta * TD_error
            self.confidence_table[(n, dest)][action] += self.eta * (next_conf - old_conf)
            # counteract the effect of confidence_decay()
            self.confidence_table[(n, dest)][action] /= self.decay
        return TD_error

    def confidence_decay(self):
        for table in self.confidence_table.values():
            for action in table:
                table[action] *= self.decay

    def update_epsilon(self):
        super().update_epsilon()
        self.confidence_decay()

class PolicyGradient(object):

    def __init__(self, network,  setting, add_entropy=False):
        self.config = {
            "learning_rate": setting['AGENT']['learning_rate'],
            "epsilon": setting['AGENT']['epsilon'],
            "epsilon_min": setting['AGENT']['epsilon_min'],
            "discount": setting['AGENT']['gamma_for_next_q_val'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "update_epsilon": False,
        }
        self.add_entropy = add_entropy
        self.theta = self.generateTheta(network)

    def generateTheta(self, network):
        theta = {}
        available_destinations = network.available_destinations
        self.available_sources = np.arange(network.numBS)
        for currpos in self.available_sources:
            nlist = list(network._network.neighbors(currpos))

            for dest in available_destinations:
                theta[(currpos, dest)] = np.zeros((network.numBS + network.numUsers))
                for t_dest in range(network.numBS, network.numUsers+network.numBS):
                    theta[(currpos,dest)][t_dest] = -np.infty if t_dest != dest else np.infty
                # if dest in nlist:
                #     for idx, action in enumerate(nlist):
                #         if action == dest:
                #             theta[(currpos, dest)][idx] = np.infty
                #         else:
                #             theta[(currpos, dest)][idx] = -np.infty
                #         action_mapping[(currpos, dest)][idx] = action
                #         inverse_action_mapping[(currpos, dest)][action] = idx
                # else:
                #     for idx, action in enumerate(nlist):
                #         try:
                #             path = nx.shortest_path_length(network._network, action, dest)
                #             theta[(currpos, dest)][idx] = 0
                #         except nx.NetworkXNoPath:
                #             theta[(currpos, dest)][idx] = -np.infty
                #         action_mapping[(currpos, dest)][idx] = action
                #         inverse_action_mapping[(currpos, dest)][action] = idx

            # for dest in available_destinations:
            #     theta[(currpos, dest)] = np.zeros((network.numBS))
            #     action_mapping[(currpos, dest)] = {}
            #     inverse_action_mapping[(currpos, dest)] = {}
            #     if dest in nlist:
            #         for idx, action in enumerate(nlist):
            #             if action == dest:
            #                 theta[(currpos, dest)][idx] = np.infty
            #             else:
            #                 theta[(currpos, dest)][idx] = -np.infty
            #             action_mapping[(currpos, dest)][idx] = action
            #             inverse_action_mapping[(currpos, dest)][action] = idx
            #     else:
            #         for idx, action in enumerate(nlist):
            #             try:
            #                 path = nx.shortest_path_length(network._network, action, dest)
            #                 theta[(currpos, dest)][idx] = 0
            #             except nx.NetworkXNoPath:
            #                 theta[(currpos, dest)][idx] = -np.infty
            #             action_mapping[(currpos, dest)][idx] = action
            #             inverse_action_mapping[(currpos, dest)][action] = idx

        for dest in available_destinations:
            theta[(dest, dest)] = {None: 0}
        return theta

    def _softmax(self, vec):
        if np.infty in vec:
            idx = np.where(vec==np.infty)
            return np.eye(vec.shape[0])[idx].flatten()
        e_theta = np.exp(vec)
        return e_theta/e_theta.sum()

    def act(self, state, neighbors):
        """ choose returns the choice following weighted random sample """
        """ Unpack the state """
        currPos, dest = state
        actions = [action for action in neighbors if (np.isfinite(self.theta[state][action]) or np.isposinf(self.theta[state][action]))]
        temp = [self.theta[state][action] for action in neighbors if (np.isfinite(self.theta[state][action]) or np.isposinf(self.theta[state][action]))]

        """ checks if the packet's current node has any available neighbors """
        if len(actions) == 0:
            next_step = None
        elif len(actions) == 1:
            next_step = actions[0]
        else:
            next_step = np.random.choice(actions, p=(self._softmax(self.theta[state][actions])))
        return next_step

    def _gradient(self, state, action, neighbors):
        """ gradient returns a vector with length of neighbors of source """
        gradient = -self._softmax(self.theta[state][neighbors])
        action_idx = np.argwhere(neighbors == action)
        gradient[action_idx] += 1
        return gradient

    def learn(self, state, action, td_error, neighbors):
        gradient = self._gradient(state, action, neighbors)
        self.theta[state][neighbors] += 0.01 * gradient * td_error

    def _update_entropy(self, reward, softmax):
        return reward - 0.1 * (softmax * np.log2(softmax)).sum()

    ''' Saves the agent's table to a pickle file for future usage '''

    def save_agent(self, path):
        pd.to_pickle(self.theta, path + "data/policy.pkl")

    ''' Load the agent's table from a pickle file for restoring a trained agent's capabilities '''

    def load_agent(self, path):
        self.theta = pd.read_pickle(path + "data/policy.pkl")

class HybridQ(PolicyGradient, QAgent):
    def __init__(self, network, setting, add_entropy=False, state_space=1, device=None):
        PolicyGradient.__init__(self, network, add_entropy=add_entropy, setting=setting)
        QAgent.__init__(self, network, setting, state_space, device)

    def learn(self, state, action, reward, done, neighbors, next_neighbors):
        softmax = self._softmax(self.theta[state])
        if self.add_entropy:
            reward = self._update_entropy(reward, softmax)
        ''' Q-Learning Algorithm '''
        td_error, _ = QAgent.learn(self, state, action, reward, done, neighbors, next_neighbors)
        if td_error is not None:
            ''' Actor Critic Algorithm '''

            biased_td_error = td_error + self.q[state][action] - max(self.q[state].values())
            PolicyGradient.learn(self, state, action, td_error=biased_td_error, neighbors=neighbors)
        return td_error, 32

    def act(self, state, neighbors):
        return PolicyGradient.act(self, state, neighbors)

    def save_agent(self, path):
        PolicyGradient.save_agent(self, path)
        QAgent.save_agent(self, path)

    ''' Load the agent's table from a pickle file for restoring a trained agent's capabilities '''
    def load_agent(self, path):
        PolicyGradient.load_agent(self, path)
        QAgent.load_agent(self, path)

class HybridCQ(PolicyGradient, CQAgent):
    def __init__(self, network, setting, add_entropy=False, state_space=1, device=None):
        PolicyGradient.__init__(self, network, add_entropy=add_entropy, setting=setting)
        CQAgent.__init__(self, network, setting, state_space, device)

    def learn(self, state, action, reward, done):
        softmax = self._softmax(self.theta[state])
        if self.add_entropy:
            reward = self._update_entropy(reward, softmax)
        ''' Q-Learning Algorithm '''
        td_error = CQAgent.learn(self, state, action, reward, done)
        if td_error is not None:
            ''' Actor Critic Algorithm '''
            PolicyGradient.learn(self, state, action, td_error=td_error)
        return td_error

    def act(self, state, neighbors):
        return PolicyGradient.act(self,state,neighbors)

    def save_agent(self, path):
        PolicyGradient.save_agent(self.theta, path)
        CQAgent.save_agent(self.q,path)

    ''' Load the agent's table from a pickle file for restoring a trained agent's capabilities '''

    def load_agent(self, path):
        PolicyGradient.load_agent(self.theta, path)
        CQAgent.load_agent(self.q, path)

class Backpressure_agent():
    def __init__(self, dynetwork, setting, state_dim, device):
        self.network = dynetwork._network
        self.dynetwork = dynetwork
        self.numNodes = len(self.network)
        self.numBs = setting["NETWORK"]["number Basestation"]
        self.numDest = setting["NETWORK"]["number Basestation"] if setting["NETWORK"]["number user"] == 0 else setting["NETWORK"]["number user"]
        self.DestOffset = self.numBs if setting["NETWORK"]["number user"] != 0 else 0
        ''' Use this function to set up the related backpressure tables'''
        self.generate_agents_table()

    def generate_agents_table(self):
        '''
        Initialize working tables to work with as part of this algorithm implemention
        '''
        self.Gamma = {}
        self.queue_backlog = {}
        self.opt_commodity = {}
        self.W = {}
        self.neighbors_to_network_mapping = {}
        return

    def extract_Buffer_Histogram(self, index):
        '''
        Calculate the current Node buffer status and divide his buffer in to Num Nodes different buffers
        '''
        # Set all the buffers to be full
        histogram = np.ones((self.numDest)) * np.infty
        if index < self.numBs:
            # This case our Node is a base station
            histogram = np.zeros((self.numDest))
            for packetInstance in self.network.nodes[index]['sending_queue']:
                endPos = packetInstance.get_endPos() - self.DestOffset
                histogram[endPos] += 1
        else:
            # This case our Node is a user
            histogram[index - self.DestOffset] = 0
        return histogram

    def find_optimal_commodity(self, currPos, currNeighbors):
        def handle_queue_diffrention(buffer, neighbor):
            neighbor_buffer = self.extract_Buffer_Histogram(neighbor)
            diff = buffer - neighbor_buffer
            if neighbor > self.numBs:
                relative_idx = neighbor - self.DestOffset
                if buffer[relative_idx] != 0:
                    diff[relative_idx] += 1
            return diff

        '''
        Find the optimal commodity and queue backlog to each neighbor
        '''
        current_Buffer_status = self.extract_Buffer_Histogram(currPos)
        self.queue_backlog[currPos] = np.zeros((len(currNeighbors), self.numDest))
        self.opt_commodity[currPos] = np.zeros((len(currNeighbors)))
        self.neighbors_to_network_mapping[currPos] = {}

        for index, neighbor in enumerate(currNeighbors):
            self.neighbors_to_network_mapping[currPos][neighbor] = index
            # Calculate the difference between the neighbor queues and the current position queues
            self.queue_backlog[currPos][index] = handle_queue_diffrention(current_Buffer_status, neighbor)
            # Mask the current infinite results from the queue backlog
            masked_queue_backlog = np.ma.array(self.queue_backlog[currPos][index], mask=~np.isfinite(self.queue_backlog[currPos][index]))
            # Find the optimal commodity to send to this neighbor based over the current backlog difference in a greedy fashion
            self.opt_commodity[currPos][index] = np.ma.argmax(masked_queue_backlog, fill_value=-np.inf)

    def calculate_weight(self,currPos, currNeighbors):
        '''
        Calculate weight of each neighbor based over the queue backlog
        '''
        self.W[currPos] = np.zeros((len(currNeighbors)))
        for index in range(len(currNeighbors)):
            # Calculate the weight of each neighbor based over the following equation
            # W[neighbor] = max(Q_diff[neighbor][optimal_commodity], 0)
            # Where optimal commodity is the commodity with the maximal queue difference
            self.W[currPos][index] = np.max(self.queue_backlog[currPos][index][int(self.opt_commodity[currPos][index])], 0)

    def solve_optimization_problem(self, currPos, currNeighbors):
        '''
        Solves the Back-Pressure optimization problem
        '''
        self.Gamma[currPos] = np.zeros((len(currNeighbors), self.numNodes))
        for rowIdx, neighborIdx in enumerate(currNeighbors):
            # Generate adjacency matrix for currPos
            self.Gamma[currPos][rowIdx][neighborIdx] = 1

        # We seek to find the maximal choice for the following summation rows
        optimization_problem = np.matmul(self.W[currPos], self.Gamma[currPos])
        neighbor_decision = np.argmax(optimization_problem)
        if np.max(optimization_problem) <= 0:
            return None, None
        else:
            chosen_destination = int(self.opt_commodity[currPos][self.neighbors_to_network_mapping[currPos][neighbor_decision]]) + self.DestOffset

        if chosen_destination == currPos:
            # In case that there is no other destination and we rather send to our self, return None
            return None, None

        # Extract the packet from the corresponding queue
        packet = None
        for currPacket in self.network.nodes[currPos]['sending_queue']:
            if currPacket.get_endPos() == chosen_destination:
                packet = currPacket.get_index()
                break

        if packet is None:
            raise Exception('Invalid Decision, Empty commodity')

        return neighbor_decision, packet

    def act(self, dynetwork, currPos, currNeighbors):
        self.dynetwork = dynetwork
        self.network = dynetwork._network
        self.find_optimal_commodity(currPos, currNeighbors)
        self.calculate_weight(currPos, currNeighbors)
        action, packet = self.solve_optimization_problem(currPos, currNeighbors)
        return action, packet

class RandomAgent(object):
    '''
    Class contains functions:
    generate_q_table: initialize Q-table
    act: returns which next node to send packet to
    learn: update Q-table after receiving corresponding rewards
    update_epsilon: update the exploration rate
    save_agent: Saves the agent's table to a pickle file for future usage
    load_agent: Load the agent's table from a pickle file for restoring a trained agent's capabilities
    '''

    def __init__(self, dynetwork, setting, state_dim, device):
        """
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values

        """
        self.config = {
            "learning_rate": setting['AGENT']['learning_rate'],
            "epsilon": setting['AGENT']['epsilon'],
            "epsilon_min": setting['AGENT']['epsilon_min'],
            "discount": setting['AGENT']['gamma_for_next_q_val'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "update_epsilon": False,
        }
        self.q = self.generate_q_table(dynetwork)

    ''' Use this function to set up the q-table'''

    def generate_q_table(self, dynetwork):
        q_table = {}
        available_destinations = dynetwork.available_destinations
        self.available_sources = np.arange(dynetwork.numBS)
        for currpos in self.available_sources:
            nlist = list(dynetwork._network.neighbors(currpos))
            for dest in available_destinations:
                q_table[(currpos, dest)] = {}
                for action in nlist:
                    ''' Initialize using Shortest Path '''
                    try:
                        q_table[(currpos, dest)][action] = nx.shortest_path_length(dynetwork._network, action, dest)
                    except nx.NetworkXNoPath:
                        q_table[(currpos, dest)][action] = -np.infty
        for dest in available_destinations:
            q_table[(dest, dest)] = {None: 0}
        return q_table

    '''Returns action for a given state by following the greedy policy. '''

    def act(self, state, neighbor):
        """ checks if the packet's current node has any available neighbors """
        available_guess = [n for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])]
        if not bool(available_guess):
            next_step = None
        else:
            next_step = random.choice(available_guess)  # Explore action space
        return next_step

class abstractDQNAgent(object):
    def __init__(self, dynetwork, setting, state_dim, device):
        self.config = {
            "nodes": dynetwork.numBS,
            "epsilon": setting['AGENT']['epsilon'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "batch_size": setting['DQN']['memory_batch_size'],
            "gamma": setting['AGENT']['gamma_for_next_q_val'],
            "TAU": setting['AGENT']['tau'],

            "update_less": setting['DQN']['optimize_per_episode'],
            "sample_memory": setting['AGENT']['use_random_sample_memory'],
            "recent_memory": setting['AGENT']['use_most_recent_memory'],
            "priority_memory": setting['AGENT']['use_priority_memory'],

            "update_epsilon": False,
            "update_models": torch.zeros([1, dynetwork.numBS], dtype=torch.bool),
            "entered": 0,
        }
        self.adjacency = dynetwork.adjacency_matrix
        self.state_dim = state_dim
        self.dqn = []
        self.device = device
        self.init_neural_networks(setting)
        self.selection_mask = torch.zeros((1, setting["NETWORK"]["number user"] + setting["NETWORK"]["number Basestation"]), dtype=torch.bool)

    def init_neural_networks(self, setting):
        raise Exception('Abstract class shall not be initalize by its own.')

    '''Update the target neural network to match the policy neural network'''
    def update_target_weights(self):
        # TAU = self.config["TAU"]
        # for target_param, local_param in zip(self.dqn.target_net.parameters(), self.dqn.policy_net.parameters()):
        #     target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)
        self.dqn.target_net.load_state_dict(self.dqn.policy_net.state_dict())

    def copy_target_weights(self):
        self.dqn.target_net.load_state_dict(self.dqn.policy_net.state_dict())

    def update_epsilon(self):
        self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']

    '''
        Updates replay memory with current experience
        and takes sample of previous experience
    '''
    def learn(self):
        ''' skip if no valid action or no reward is provided '''
        ''' check if our memory bank has sufficient memories to sample from '''
        if self.dqn.replay_memory.can_provide_sample(self.config['batch_size']):
            '''check which type of memories to pull'''
            if self.config['sample_memory']:
                experiences = self.dqn.replay_memory.sample(self.config['batch_size'])

            elif self.config['recent_memory']:
                experiences = self.dqn.replay_memory.take_recent(self.config['batch_size'])

            elif self.config['priority_memory']:
                experiences, experiences_idx = self.dqn.replay_memory.take_priority(self.config['batch_size'])
            else:
                raise Exception('Invalid Memory usage')

            states, actions, next_states, rewards, dones, destinations = self.extract_tensors(experiences)

            '''extract values from experiences'''
            current_q_values = self.get_current_QVal(self.dqn.policy_net, states, actions)

            next_q_values = self.get_next_QVal(self.dqn.policy_net, self.dqn.target_net, next_states, actions, dones, destinations)

            target_q_values = (next_q_values * self.config['gamma']) + rewards

            '''update priority memory's probability'''
            if self.config['priority_memory']:
                self.dqn.replay_memory.update_priorities(experiences_idx, current_q_values, torch.transpose(target_q_values, 0, 1))

            '''backpropagation to update neural network'''
            loss = F.mse_loss(current_q_values, torch.transpose(target_q_values, 0, 1))
            self.dqn.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dqn.policy_net.parameters(), 5)
            self.dqn.optimizer.step()
            return torch.mean(current_q_values - target_q_values).item()

    ''' helper function to extract values from our stored experiences'''

    def extract_tensors(self, experiences):
        states = torch.cat(tuple(exps[0] for exps in experiences)).to(self.device)
        actions = torch.cat(tuple(torch.tensor([exps[1]]) for exps in experiences)).to(self.device)
        next_states = torch.cat(tuple(exps[2] for exps in experiences)).to(self.device)
        rewards = torch.cat(tuple(torch.tensor([exps[3]]) for exps in experiences)).to(self.device)
        dones = torch.cat(tuple(torch.tensor([exps[4]]) for exps in experiences)).to(self.device)
        destinations = torch.cat(tuple(torch.tensor([exps[5]]) for exps in experiences)).to(self.device)
        return (states, actions, next_states, rewards, dones, destinations)

    def save_agent(self, path):
        raise Exception('Abstract class shall not be initalize by its own.')

    def load_agent(self, path):
        raise Exception('Abstract class shall not be initalize by its own.')

    ''' helper function to obtain the Q-val of current state'''

    def get_current_QVal(self, policy_net, states, actions):
        return policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))

'''
The following classes implements a learning agent based over MLP network and its hyper-parameters
'''
class CentralizedDQNAgent(abstractDQNAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)


    ''' helper function to generate the action-value function approximation for each agent '''
    def init_neural_networks(self, setting):
        ''' Initialize all neural networks with one neural network initialized for each node in the network. '''
        self.dqn = DQN(0, self.state_dim, setting=setting, device=self.device)
        return


    """
    act is a function that gives a packet the next best action (if possible)
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank
    """
    def act(self, state, neighbor, dest):
        ''' We will either random walk or reference Deep Q function approximation with probability epsilon '''
        available_guess = [n for n in neighbor if (n < self.config["nodes"] or n == dest)]
        if random.uniform(0, 1) < self.config['epsilon']:
            if not bool(available_guess):
                # checks if the packet's current node has any available neighbors
                return None
            else:
                # Explore action space
                next_step = random.choice(available_guess)
        else:
            if not bool(neighbor):
                return None
            else:
                ''' obtains the next best neighbor to move the packet from its current node by referencing our neural network '''
                with torch.no_grad():
                    state = state.to(self.device)
                    qvals = self.dqn.policy_net(state.float()).cpu()
                    next_step_idx = qvals[:, available_guess].argmax().item()
                    next_step = available_guess[next_step_idx]
        return next_step

    ''' helper function to obtain the Q-val of the next state'''
    def get_next_QVal(self, policy_net, target_net, next_states, actions, dones, destinations):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''
        target_q_val_est = target_net(next_states.float())
        curr_q_val_est = policy_net(next_states.float())

        ''' initialize zero value vectors '''
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)

        ''' update non-terminal state with Q value '''
        for idx in range(values.size()[1]):
            if not dones[idx]:
                # Extract the next node neighbors from the adjacency matrix
                adjs = self.adjacency[actions[idx]][0]
                next_hops = np.argwhere(adjs == 1)[:, 1]
                available_guess = [n for n in next_hops if (n < self.config["nodes"] or n == destinations[idx])]
                # Extract the next node target Q values
                next_node_target_values = target_q_val_est[idx, :].view(1, -1)
                next_node_curr_values = curr_q_val_est[idx, :].view(1, -1)
                self.selection_mask[0, available_guess] = True
                # Follow the greedy policy only from the next node available neighbors using double DQN method
                action_idx = torch.argmax(next_node_curr_values[self.selection_mask])
                values[0, idx] = next_node_target_values[self.selection_mask][action_idx].detach()
                self.selection_mask[0, available_guess] = False

        return values

    ''' helper function to save the agents weights '''
    def save_agent(self, path):
        torch.save(self.dqn.target_net.state_dict(), path+f'data/agent')

    ''' helper function to load the agents weights '''
    def load_agent(self, path):
        self.dqn.target_net.load_state_dict(torch.load(path+f'data/agent'))
        self.dqn.target_net.eval()
        self.dqn.policy_net.load_state_dict(torch.load(path + f'data/agent'))

        for param in self.dqn.policy_net.layer1.parameters():
            param.requires_grad = False
        # for param in self.dqn.policy_net.layer2.parameters():
        #     param.requires_grad = False

class DecentralizedDQNAgent(abstractDQNAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)

    def init_neural_networks(self,setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        for i in range(self.config["nodes"]):
            temp_dqn = DQN(i, self.state_dim, setting=setting, device=self.device)
            self.dqn.append(temp_dqn)
        return

    def act(self, agentIdx, state, neighbor, dest):
        ''' We will either random walk or reference Deep Q function approximation with probability epsilon '''
        available_guess = [n for n in neighbor if (n < self.config["nodes"] or n == dest)]
        if random.uniform(0, 1) < self.config['epsilon']:
            if not bool(available_guess):
                # checks if the packet's current node has any available neighbors
                return None
            else:
                # Explore action space
                next_step = random.choice(available_guess)
        else:
            if not bool(neighbor):
                return None
            else:
                ''' obtains the next best neighbor to move the packet from its current node by referencing our neural network '''
                with torch.no_grad():
                    state = state.to(self.device)
                    qvals = self.dqn[agentIdx].policy_net(state.float()).cpu()
                    next_step_idx = qvals[:, available_guess].argmax().item()
                    next_step = available_guess[next_step_idx]
        return next_step
    '''
        Updates replay memory with current experience
        and takes sample of previous experience
    '''
    def learn(self):
        ''' skip if no valid action or no reward is provided '''
        for nn in self.dqn:
            ''' check if our memory bank has sufficient memories to sample from '''
            if nn.replay_memory.can_provide_sample(self.config['batch_size']):

                '''check which type of memories to pull'''
                if self.config['sample_memory']:
                    experiences = nn.replay_memory.sample(self.config['batch_size'])

                elif self.config['recent_memory']:
                    experiences = nn.replay_memory.take_recent(self.config['batch_size'])

                elif self.config['priority_memory']:
                    experiences, experiences_idx = nn.replay_memory.take_priority(self.config['batch_size'])
                else:
                    raise Exception('Invalid Memory usage')

                states, actions, next_states, rewards, dones, destinations = self.extract_tensors(experiences)

                '''extract values from experiences'''
                current_q_values = self.get_current_QVal(nn.policy_net, states, actions, destinations)

                next_q_values = self.get_next_QVal(next_states, actions, dones)

                '''
                Normalize the rewards
                '''
                target_q_values = (next_q_values * self.config['gamma']) + rewards

                '''update priority memory's probability'''
                if self.config['priority_memory']:
                    nn.replay_memory.update_priorities(experiences_idx, current_q_values, torch.transpose(target_q_values, 0, 1))

                '''backpropagation to update neural network'''
                loss = F.mse_loss(current_q_values, torch.transpose(target_q_values, 0, 1))
                nn.optimizer.zero_grad()
                loss.backward()
                nn.optimizer.step()
                return torch.mean(current_q_values - target_q_values).item()

    ''' helper function to obtain the Q-val of the next state'''

    def get_next_QVal(self, next_states, actions, dones):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''

        ''' initialize zero value vectors '''
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)

        ''' update non-terminal state with Q value '''
        with torch.no_grad():
            for idx in range(batch_size):
                if not dones[idx]:
                    next_node = actions[idx]
                    # Extract the next node neighbors from the adjacency matrix
                    adjs = self.adjacency[next_node][0]
                    adjs = (adjs == 1)

                    # Get the next node networks
                    policy_network = self.dqn[next_node].policy_net
                    target_network = self.dqn[next_node].target_net

                    # Get the next node state estimation
                    curr_q_val_est = policy_network(next_states[idx]).unsqueeze(0)
                    target_q_val_est = target_network(next_states[idx]).unsqueeze(0)

                    # Follow the greedy policy only from the next node available neighbors using double DQN method
                    action_idx = torch.argmax(curr_q_val_est[adjs])
                    values[0, idx] = target_q_val_est[adjs][action_idx]

        return values

    def update_target_weights(self):
        TAU = self.config["TAU"]
        for nn in self.dqn:
            for target_param, local_param in zip(nn.target_net.parameters(), nn.policy_net.parameters()):
                target_param.data.copy_(TAU*local_param.data + (1.0-TAU)*target_param.data)

    def copy_target_weights(self):
        for nn in self.dqn:
            nn.target_net.load_state_dict(nn.policy_net.state_dict())

    ''' helper function to save the agents weights '''

    def save_agent(self, path):
        for nn in self.dqn:
            torch.save(nn.target_net.state_dict(), path+f'data/src_{nn.ID}')

    ''' helper function to load the agents weights '''

    def load_agent(self, path):
        for nn in self.dqn:
            nn.target_net.load_state_dict(torch.load(path+f'data/src_{nn.ID}'))
            nn.target_net.eval()
            nn.policy_net.load_state_dict(torch.load(path + f'data/src_{nn.ID}'))
            nn.policy_net.eval()

class RelationalDQNAgent(abstractDQNAgent):
    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.selection_mask = torch.zeros((1, setting["NETWORK"]["number Basestation"]), dtype=torch.bool)

    def init_neural_networks(self, setting):
        ''' Initialize all neural networks with one neural network initialized for each node in the network. '''
        self.dqn = RelationalDQN(self.state_dim, setting=setting, device=self.device)
        return

    '''Update the target neural network to match the policy neural network'''
    def update_target_weights(self):
        self.dqn.target_net.load_state_dict(self.dqn.policy_net.state_dict())

    def copy_target_weights(self):
        self.dqn.target_net.load_state_dict(self.dqn.policy_net.state_dict())

    def update_epsilon(self):
        self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']

    '''
        Updates replay memory with current experience
        and takes sample of previous experience
    '''
    def learn(self):
        ''' skip if no valid action or no reward is provided '''
        ''' check if our memory bank has sufficient memories to sample from '''
        if self.dqn.replay_memory.can_provide_sample(self.config['batch_size']):

            experiences = self.dqn.replay_memory.sample(self.config['batch_size'])

            states, actions, next_states, rewards, dones, destinations = self.extract_tensors(experiences)

            '''extract values from experiences'''
            current_q_values = self.get_current_QVal(self.dqn.policy_net, states, actions)

            next_q_values = self.get_next_QVal(self.dqn.policy_net, self.dqn.target_net, next_states, actions, dones, destinations)

            target_q_values = (next_q_values * self.config['gamma']) + rewards

            '''backpropagation to update neural network'''
            loss = F.mse_loss(current_q_values, torch.transpose(target_q_values, 0, 1))
            self.dqn.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.dqn.policy_net.parameters(), 5)
            self.dqn.optimizer.step()
            return torch.mean(current_q_values - target_q_values).item()

    ''' helper function to extract values from our stored experiences'''

    def extract_tensors(self, experiences):
        states = torch.cat(tuple(exps[0] for exps in experiences)).to(self.device)
        actions = torch.cat(tuple(torch.tensor([exps[1]]) for exps in experiences)).to(self.device)
        next_states = torch.cat(tuple(exps[2] for exps in experiences)).to(self.device)
        rewards = torch.cat(tuple(torch.tensor([exps[3]]) for exps in experiences)).to(self.device)
        dones = torch.cat(tuple(torch.tensor([exps[4]]) for exps in experiences)).to(self.device)
        destinations = torch.cat(tuple(torch.tensor([exps[5]]) for exps in experiences)).to(self.device)
        return (states, actions, next_states, rewards, dones, destinations)

    def save_agent(self, path):
        torch.save(self.dqn.target_net.state_dict(), path + f'data/agent')

    ''' helper function to load the agents weights '''

    def load_agent(self, path):
        self.dqn.target_net.load_state_dict(torch.load(path + f'data/agent'))
        self.dqn.target_net.eval()
        self.dqn.policy_net.load_state_dict(torch.load(path + f'data/agent'))
        self.dqn.policy_net.eval()

    ''' helper function to obtain the Q-val of current state'''

    def get_current_QVal(self, policy_net, states, actions):
        return policy_net(states.float()).gather(dim=0, index=actions.unsqueeze(-1))

    def get_next_QVal(self, policy_net, target_net, next_states, actions, dones, destinations):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''
        target_q_val_est = target_net(next_states.float())
        curr_q_val_est = policy_net(next_states.float())

        ''' initialize zero value vectors '''
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)

        ''' update non-terminal state with Q value '''
        for idx in range(values.size()[1]):
            if not dones[idx]:
                # Extract the next node neighbors from the adjacency matrix
                adjs = self.adjacency[actions[idx]][0]
                next_hops = np.argwhere(adjs == 1)[:, 1]
                if destinations[idx] in next_hops:
                    available_guess = [destinations[idx]]
                else:
                    available_guess = [n for n in next_hops if n < self.config["nodes"]]
                # Extract the next node target Q values
                next_node_target_values = target_q_val_est[idx, :].view(1, -1)
                next_node_curr_values = curr_q_val_est[idx, :].view(1, -1)
                self.selection_mask[0, available_guess] = True
                # Follow the greedy policy only from the next node available neighbors using double DQN method
                action_idx = torch.argmax(next_node_curr_values[self.selection_mask])
                values[0, idx] = next_node_target_values[self.selection_mask][action_idx].detach()
                self.selection_mask[0, available_guess] = False

        return values

    def act(self, state, neighbor, dest):
        ''' We will either random walk or reference Deep Q function approximation with probability epsilon '''
        if dest in neighbor:
            return dest

        available_guess = [n for n in neighbor if n < self.config["nodes"]]
        if random.uniform(0, 1) < self.config['epsilon']:
            if not bool(available_guess):
                # checks if the packet's current node has any available neighbors
                return None
            else:
                # Explore action space
                next_step = random.choice(available_guess)
        else:
            if not bool(neighbor):
                return None
            else:
                ''' obtains the next best neighbor to move the packet from its current node by referencing our neural network '''
                with torch.no_grad():
                    state = state.to(self.device)
                    qvals = self.dqn.policy_net(state.float()).cpu()
                    next_step_idx = qvals[:, available_guess].argmax().item()
                    next_step = available_guess[next_step_idx]
        return next_step

class abstractA2CAgent(object):
    def __init__(self, dynetwork, setting, state_dim, device):
        self.config = {
            "nodes": dynetwork.numBS,
            "epsilon": setting['AGENT']['epsilon'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "batch_size": setting['DQN']['memory_batch_size'],
            "gamma": setting['AGENT']['gamma_for_next_q_val'],
            "TAU": setting['AGENT']['tau'],

            "update_less": setting['DQN']['optimize_per_episode'],
            "sample_memory": setting['AGENT']['use_random_sample_memory'],
            "recent_memory": setting['AGENT']['use_most_recent_memory'],
            "priority_memory": setting['AGENT']['use_priority_memory'],

            "update_epsilon": False,
            "update_models": torch.zeros([1, dynetwork.numBS], dtype=torch.bool),
            "entered": 0,
        }
        self.adjacency = dynetwork.adjacency_matrix
        self.state_dim = state_dim
        self.numNodes = dynetwork.numBS + dynetwork.numUsers
        self.actor = None
        self.critic = None
        self.device = device


    '''Update the target neural network to match the policy neural network'''

    def update_target_weights(self):
        pass

    def copy_target_weights(self):
        pass

    def update_epsilon(self):
        pass

    """
    act is a function that gives a packet the next best action (if possible)
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank
    """

    def masked_softmax(self, x, mask, **kwargs):
        exps = torch.exp(x)
        masked_exps = exps * torch.tensor(mask)
        masked_sums = masked_exps.sum(-1, keepdim=True)
        res = (masked_exps / masked_sums)
        if torch.isnan(res).any():
            res.detach()
            exps = 1 / torch.exp(x.T - x).sum(axis=0)
            masked_exps = exps * torch.tensor(mask)
            masked_sums = masked_exps.sum(-1, keepdim=True)
            res = (masked_exps / masked_sums)
        return res

        # x_masked = x.clone().squeeze(0)
        # x_masked[mask == 0] = -float("inf")
        # return torch.softmax(x_masked, **kwargs)

    ''' helper function to extract values from our stored experiences'''
    def update_central_controller_weights(self):
        pass

    def extract_tensors(self, experiences):
        states = torch.cat(tuple(exps[0] for exps in experiences)).to(self.device)
        actions = torch.cat(tuple(torch.tensor([exps[1]]) for exps in experiences)).to(self.device)
        rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in experiences)).to(self.device)
        next_states = torch.cat(tuple(exps[3] for exps in experiences)).to(self.device)
        dones = torch.cat(tuple(torch.tensor([exps[4]]) for exps in experiences)).to(self.device)
        neighbors = tuple(exps[5] for exps in experiences)
        destinations = torch.cat(tuple(torch.tensor([exps[6]]) for exps in experiences)).to(self.device)
        return (states, actions, next_states, rewards, dones, neighbors, destinations)

# class DoubleQAgent(object):
#     '''
#         Class contains functions:
#         generate_q_table: initialize Q-table
#         act: returns which next node to send packet to
#         learn: update Q-table after receiving corresponding rewards
#         update_epsilon: update the exploration rate
#         save_agent: Saves the agent's table to a pickle file for future usage
#         load_agent: Load the agent's table from a pickle file for restoring a trained agent's capabilities
#         '''
#
#     def __init__(self, dynetwork, setting, state_dim, device):
#         """
#         learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
#         epsilon: probability that packets move randomly, instead of referencing routing policy
#         discount: Degree to which we wish to maximize future rewards, value between (0,1)
#         decay_rate: decays epsilon
#         update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
#         self.q: stores q-values
#
#         """
#         self.config = {
#             "learning_rate": setting['AGENT']['learning_rate'],
#             "epsilon": setting['AGENT']['epsilon'],
#             "epsilon_min": setting['AGENT']['epsilon_min'],
#             "discount": setting['AGENT']['gamma_for_next_q_val'],
#             "decay_rate": setting['AGENT']['decay_epsilon_rate'],
#             "update_epsilon": False,
#         }
#         self.q = self.generate_q_table(dynetwork)
#         self.double_q = self.generate_q_table(dynetwork)
#
#     ''' Use this function to set up the q-table'''
#
#     def generate_q_table(self, dynetwork):
#         q_table = {}
#         available_destinations = dynetwork.available_destinations
#         self.available_sources = np.arange(dynetwork.numBS)
#         for currpos in self.available_sources:
#             nlist = list(dynetwork._network.neighbors(currpos))
#             for dest in available_destinations:
#                 q_table[(currpos, dest)] = {}
#                 for action in nlist:
#                     # if dest == action or action in self.available_sources:
#                     #     ''' Initialize 0 Q-table except destination '''
#                     # q_table[(currpos, dest)][action] = -1000
#                     ''' Initialize using Shortest Path '''
#                     try:
#                         q_table[(currpos, dest)][action] = nx.shortest_path_length(dynetwork._network, action, dest)
#                     except nx.NetworkXNoPath:
#                         q_table[(currpos, dest)][action] = -np.infty
#         for dest in available_destinations:
#             q_table[(dest, dest)] = {None: 0}
#         return q_table
#
#     '''Returns action for a given state by following the greedy policy. '''
#
#     def act(self, state, neighbor):
#         ''' We will either random walk or reference Q-table with probability epsilon '''
#         if random.uniform(0, 1) < max(self.config['epsilon'], self.config["epsilon_min"]):
#             """ checks if the packet's current node has any available neighbors """
#             available_guess = [n for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])]
#             if not bool(available_guess):
#                 return None
#             else:
#                 next_step = random.choice(available_guess)  # Explore action space
#         else:
#             temp_neighbor_dict = {n: 0.5*(self.q[state][n]+self.double_q[state][n]) for n in self.q[state] if n in neighbor and np.isfinite(self.q[state][n])}
#             """ checks if the packet's current node has any available neighbors """
#             if not bool(temp_neighbor_dict):
#                 return None
#             else:
#                 next_step = max(temp_neighbor_dict, key=temp_neighbor_dict.get)
#         return next_step
#
#     ''' update the exploration rate in a decaying manner'''
#
#     def update_epsilon(self):
#         self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']
#
#     """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """
#
#     def learn(self, current_event, action, reward, done):
#         if (action == None) or (reward == None):
#             TD_error = None
#         else:
#             n, dest = current_event
#             updateIndex = np.random.choice(2)
#             if updateIndex == 0:
#                 max_q_idx = np.argmax(self.q[(action, dest)].values())
#                 max_q_idx = list(self.q[(action, dest)].keys())[max_q_idx]
#                 """ Q learning algorithm """
#                 TD_error = (reward + self.config["discount"] * self.double_q[(action, dest)][max_q_idx] * (1 - done) - self.q[(n, dest)][action])
#                 self.q[(n, dest)][action] = self.q[(n, dest)][action] + (self.config["learning_rate"]) * TD_error
#             else:
#                 max_q_idx = np.argmax(self.double_q[(action, dest)].values())
#                 max_q_idx = list(self.double_q[(action, dest)].keys())[max_q_idx]
#                 """ Q learning algorithm """
#                 TD_error = (reward + self.config["discount"] * self.q[(action, dest)][max_q_idx] * (1 - done) -
#                             self.double_q[(n, dest)][action])
#                 self.double_q[(n, dest)][action] = self.double_q[(n, dest)][action] + (self.config["learning_rate"]) * TD_error
#
#         return TD_error
#
#     ''' Saves the agent's table to a pickle file for future usage '''
#
#     def save_agent(self, path):
#         pd.to_pickle(self.q, path + "data/agent.pkl")
#
#     ''' Load the agent's table from a pickle file for restoring a trained agent's capabilities '''
#
#     def load_agent(self, path):
#         self.q = pd.read_pickle(path + "data/agent.pkl")
#
# class DoubleFullEchoQAgent(DoubleQAgent):
#     def __init__(self, dynetwork, setting, state_dim, device):
#         super().__init__(dynetwork, setting, state_dim, device)
#         self.config['epsilon'] = 0.0
#         self.config['epsilon_min'] = 0.0
#
#     """updates q-table given current state, reward, and action where a state is a (Node, destination) pair and an action is a step to of the neighbors of the Node """
#
#     def learn(self, current_event, action, rewards, dones):
#         if (action == None) or (rewards == None):
#             TD_error = None
#         else:
#             state, neighbor_list = current_event
#             n, dest = state
#             TD_errors = np.zeros((len(neighbor_list), 1))
#
#
#             ''' Iterate through all available neighbors '''
#             for cnt, info in enumerate(zip(neighbor_list, rewards, dones)):
#                 neighbor, reward, done = info
#
#                 updateIndex = np.random.choice(2)
#                 if updateIndex == 0:
#                     max_q_idx = np.argmax(self.q[(neighbor, dest)].values())
#                     max_q_idx = list(self.q[(neighbor, dest)].keys())[max_q_idx]
#                     """ Q learning algorithm """
#                     TD_error = (reward + self.config["discount"] * self.double_q[(neighbor, dest)][max_q_idx] * (1 - done) - self.q[(n, dest)][neighbor])
#                     self.q[(n, dest)][neighbor] = self.q[(n, dest)][neighbor] + (self.config["learning_rate"]) * TD_error
#                 else:
#                     max_q_idx = np.argmax(self.double_q[(neighbor, dest)].values())
#                     max_q_idx = list(self.double_q[(neighbor, dest)].keys())[max_q_idx]
#                     """ Q learning algorithm """
#                     TD_error = (reward + self.config["discount"] * self.q[(neighbor, dest)][max_q_idx] * (1 - done) -
#                                 self.double_q[(n, dest)][neighbor])
#                     self.double_q[(n, dest)][neighbor] = self.double_q[(n, dest)][neighbor] + (self.config["learning_rate"]) * TD_error
#                 TD_errors[cnt] = TD_error
#             TD_error = np.mean(TD_errors)
#         return TD_error
# class DecentralizedDQNAgent(DQNAgent):
#     """
#
#         Initialize an instance of the agent class
#         learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
#         epsilon: probability that packets move randomly, instead of referencing routing policy
#         discount: Degree to which we wish to maximize future rewards, value between (0,1)
#         decay_rate: decays epsilon
#         update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
#         self.q: stores q-values
#         *_memory: different methods of sampling from our memory bank
#
#     """
#
#     def __init__(self, dynetwork, setting, state_dim, device):
#         super().__init__(dynetwork, setting, state_dim, device)
#
#     '''
#         Updates replay memory with current experience
#         and takes sample of previous experience
#     '''
#     def learn(self, idx, current_event, action, reward, next_state):
#         ''' skip if no valid action or no reward is provided '''
#         if (action == None) or (reward == None):
#             pass
#         else:
#             nn = self.dqn[idx]
#             ''' check if our memory bank has sufficient memories to sample from '''
#             if nn.replay_memory.can_provide_sample(self.config['batch_size']):
#
#                 '''check which type of memories to pull'''
#                 if self.config['sample_memory']:
#                     experiences = nn.replay_memory.sample(self.config['batch_size'])
#
#                 elif self.config['recent_memory']:
#                     experiences = nn.replay_memory.take_recent(self.config['batch_size'])
#
#                 elif self.config['priority_memory']:
#                     experiences, experiences_idx = nn.replay_memory.take_priority(self.config['batch_size'])
#                 else:
#                     raise Exception('Invalid Memory usage')
#
#                 states, actions, next_states, rewards, dones = self.extract_tensors(experiences)
#
#                 '''extract values from experiences'''
#                 current_q_values = self.get_current_QVal(nn.policy_net, states, actions)
#
#                 next_q_values = self.get_next_QVal_dec(next_states, actions, dones)
#
#                 '''
#                 Normalize the rewards
#                 '''
#                 target_q_values = (next_q_values * self.config['gamma']) + rewards
#
#                 '''update priority memory's probability'''
#                 if self.config['priority_memory']:
#                     nn.replay_memory.update_priorities(experiences_idx, current_q_values, torch.transpose(target_q_values, 0, 1))
#
#                 '''backpropagation to update neural network'''
#                 loss = F.mse_loss(current_q_values, torch.transpose(target_q_values, 0, 1))
#                 nn.optimizer.zero_grad()
#                 loss.backward()
#                 nn.optimizer.step()
#                 return torch.mean(current_q_values - target_q_values).item()
#
#     ''' helper function to obtain the Q-val of the next state'''
#
#     def get_next_QVal_dec(self, next_states, actions, dones):
#         ''' need to apply neighbors mask to target_net() prior to taking the maximum '''
#
#         ''' initialize zero value vectors '''
#         batch_size = next_states.shape[0]
#         values = torch.zeros(batch_size).view(1, -1).to(self.device)
#
#         ''' update non-terminal state with Q value '''
#         with torch.no_grad():
#             for idx in range(batch_size):
#                 if not dones[idx]:
#                     next_node = actions[idx]
#                     # Extract the next node neighbors from the adjacency matrix
#                     adjs = self.adjacency[next_node][0]
#                     adjs = (adjs == 1)
#
#                     # Get the next node networks
#                     policy_network = self.dqn[next_node].policy_net
#                     target_network = self.dqn[next_node].target_net
#
#                     # Get the next node state estimation
#                     curr_q_val_est = policy_network(next_states[idx]).unsqueeze(0)
#                     target_q_val_est = target_network(next_states[idx]).unsqueeze(0)
#
#                     # Follow the greedy policy only from the next node available neighbors using double DQN method
#                     action_idx = torch.argmax(curr_q_val_est[adjs])
#                     values[0, idx] = target_q_val_est[adjs][action_idx]
#
#         return values
#
#     ''' helper function to save the agents weights '''
#
#     def save_agent(self, path):
#         for nn in self.dqn:
#             torch.save(nn.target_net.state_dict(), path+f'data/src_{nn.ID}')
#
#     ''' helper function to load the agents weights '''
#
#     def load_agent(self, path):
#         for nn in self.dqn:
#             nn.target_net.load_state_dict(torch.load(path+f'data/src_{nn.ID}'))
#             nn.target_net.eval()
#             nn.policy_net.load_state_dict(torch.load(path + f'data/src_{nn.ID}'))
#             nn.policy_net.eval()

class CentralizedA2CAgent(abstractA2CAgent):
    """

            Initialize an instance of the agent class
            learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
            epsilon: probability that packets move randomly, instead of referencing routing policy
            discount: Degree to which we wish to maximize future rewards, value between (0,1)
            decay_rate: decays epsilon
            update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
            self.q: stores q-values
            *_memory: different methods of sampling from our memory bank

        """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.init_neural_networks(setting)

    ''' helper function to generate the action-value function approximation for each agent '''

    def init_neural_networks(self, setting):
        ''' Initialize all neural networks with one neural network initialized for each node in the network. '''
        self.actor = DQN(0, self.state_dim, setting=setting, device=self.device)
        self.critic = ValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
        return

    """
    act is a function that gives a packet the next best action (if possible)
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank
    """

    def act(self, state, neighbor, dest):
        ''' We will sample our next action according to our actor distribution '''
        available_guess = [n for n in neighbor if (n < self.config["nodes"] or n == dest)]
        if not bool(available_guess):
            # checks if the packet's current node has any available neighbors
            return None
        else:
            # Generate next action masking
            mask = np.zeros((1, self.numNodes))
            mask[0, available_guess] = 1
            # Calculate Policy distribution
            with torch.no_grad():
                policy = self.actor.policy_net(state)
            distribution = self.masked_softmax(policy, mask, dim=-1).unsqueeze(0)
            m = Categorical(distribution)
            next_step = m.sample().item()

        return next_step

    def learn(self, transitions):
        ''' skip if no valid action or no reward is provided '''
        ''' check if our memory bank has sufficient memories to sample from '''
        if not transitions:
            return 0, 0, None

        states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions)

        '''extract values estimations from experiences'''
        current_values = self.get_current_Val(states).squeeze()
        next_values = self.get_next_Val(next_states).squeeze()
        next_values = (~dones) * (next_values * self.config['gamma']) + rewards
        ''' Exctract Policy Estimation from Experience '''
        policy_pred = self.actor.policy_net(states)
        masks = [[1 if ((n in neighbor) and (n < self.config["nodes"] or n == dest)) else 0 for n in range(self.numNodes)] for neighbor, dest in zip(neighbors, destinations)]
        numStates = states.shape[0]
        distributions = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
        neg_log_prob = -torch.log(distributions[range(numStates), actions])
        loss_actor = (neg_log_prob * (next_values - current_values)).mean()
        loss_critic = F.mse_loss(current_values, next_values)
        loss = loss_critic + loss_actor
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        '''backpropagation to update neural network'''
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.critic.network.parameters(), 10)
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return loss_critic.item(), loss_actor.item(), rewards.sum()

    ''' helper function to obtain the Q-val of current state'''

    def get_current_Val(self, states):
        return self.critic.network(states.float())

    ''' helper function to obtain the Q-val of the next state'''

    def get_next_Val(self, next_states):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''
        return self.critic.network(next_states.float()).detach()

    ''' helper function to save the agents weights '''

    def save_agent(self, path):
        torch.save(self.critic.network.state_dict(), path + f'data/agent_critic')
        torch.save(self.actor.policy_net.state_dict(), path + f'data/agent_policy')

    ''' helper function to load the agents weights '''

    def load_agent(self, path):
        self.actor.policy_net.load_state_dict(torch.load(path + f'data/agent_policy'))
        self.actor.policy_net.eval()
        self.critic.network.load_state_dict(torch.load(path + f'data/agent_critic'))
        self.critic.network.eval()

class CentralizedPPOAgent(CentralizedA2CAgent):
    """

            Initialize an instance of the agent class
            learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
            epsilon: probability that packets move randomly, instead of referencing routing policy
            discount: Degree to which we wish to maximize future rewards, value between (0,1)
            decay_rate: decays epsilon
            update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
            self.q: stores q-values
            *_memory: different methods of sampling from our memory bank

        """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.clip_range = 0.2

    def normalize(self, var: torch.Tensor):
        return (var - var.mean()) / (var.std() + 1e-10)

    def learn(self, transitions):
        ''' skip if no valid action or no reward is provided '''
        ''' check if our memory bank has sufficient memories to sample from '''
        if not transitions:
            return 0, 0, None

        states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions)

        '''extract values estimations from experiences'''
        current_values = self.get_current_Val(states).squeeze()
        next_values = self.get_next_Val(next_states).squeeze()
        next_values = (~dones) * (next_values * self.config['gamma']) + rewards
        ''' Exctract Policy Estimation from Experience '''
        with torch.no_grad():
            policy_pred = self.actor.policy_net(states)
            masks = [[1 if ((n in neighbor) and (n < self.config["nodes"] or n == dest)) else 0 for n in range(self.numNodes)] for neighbor, dest in zip(neighbors, destinations)]
            numStates = states.shape[0]
            distributions = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
            old_log_policy = torch.log(distributions[range(numStates), actions] + 1e-10)
            sampled_normalized_advantage = self.normalize(next_values - current_values).detach()

        for epoch in range(8):
            policy_pred = self.actor.policy_net(states)
            policy_pred = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
            new_log_policy = torch.log(policy_pred[range(numStates), actions] + 1e-10)
            policy_ratio = torch.exp(new_log_policy - old_log_policy)
            clipped_ratio = policy_ratio.clamp(min=1.0 - self.clip_range, max=1.0 + self.clip_range)
            policy_loss = torch.min(policy_ratio * sampled_normalized_advantage, clipped_ratio * sampled_normalized_advantage)
            policy_loss = policy_loss.mean()
            self.actor.optimizer.zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), max_norm=10)
            self.actor.optimizer.step()

        loss_critic = F.mse_loss(current_values, next_values)
        self.critic.optimizer.zero_grad()
        '''backpropagation to update neural network'''
        loss_critic.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.network.parameters(), max_norm=10)
        self.critic.optimizer.step()

        return loss_critic.item(), policy_loss.item(), rewards.sum()

class DecentralizedA2CAgent(abstractA2CAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.critics = []
        self.actors = []
        self.init_neural_networks(setting)
        self.numAgents = len(self.actors)

    def init_neural_networks(self,setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        for i in range(self.config["nodes"]):
            actor  = DQN(i, self.state_dim, setting=setting, device=self.device)
            critic = ValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        return

    def act(self, agentIdx, state, neighbor, dest):
        ''' We will sample our next action according to our actor distribution '''
        available_guess = [n for n in neighbor if (n < self.config["nodes"] or n == dest)]
        if not bool(available_guess):
            # checks if the packet's current node has any available neighbors
            return None
        else:
            # Generate next action masking
            mask = np.zeros((1, self.numNodes))
            mask[0, available_guess] = 1
            # Calculate Policy distribution
            with torch.no_grad():
                policy = self.actors[agentIdx].policy_net(state)
            distribution = self.masked_softmax(policy, mask, dim=-1).unsqueeze(0)
            m = Categorical(distribution)
            next_step = m.sample().item()
        return next_step

    ''' helper function to obtain the Q-val of current state'''

    def learn(self, transitions):
        if not transitions:
            return 0, 0, None

        states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions)
        sources = torch.cat(tuple(torch.tensor([exps[7]]) for exps in transitions)).to(self.device)
        ''' Extract Critics and Policy Estimation from Experience '''
        current_values, policy_pred = self.get_current_Val_and_policies(states, sources)
        ''' Get the Next State estimation and calculate the return to go estimation '''
        next_values = self.get_next_Val(next_states, actions, dones).squeeze()
        next_values = (~dones) * (next_values * self.config['gamma']) + rewards
        ''' Generate Masks according to the current neighborhood '''
        masks = [[1 if ((n in neighbor) and (n < self.config["nodes"] or n == dest)) else 0 for n in range(self.numNodes)] for neighbor, dest in
                 zip(neighbors, destinations)]
        numStates = states.shape[0]
        distributions = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
        neg_log_prob = -torch.log(distributions[range(numStates), actions])
        loss_actor = (neg_log_prob * (next_values - current_values)).mean()
        loss_critic = F.mse_loss(current_values, next_values)
        loss = loss_critic + loss_actor
        for agentIdx in range(self.numAgents):
            self.critics[agentIdx].optimizer.zero_grad()
            self.actors[agentIdx].optimizer.zero_grad()
        '''backpropagation to update neural network'''
        loss.backward()
        for agentIdx in range(self.numAgents):
            torch.nn.utils.clip_grad_norm_(self.actors[agentIdx].policy_net.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(self.critics[agentIdx].network.parameters(), 10)
            self.actors[agentIdx].optimizer.step()
            self.critics[agentIdx].optimizer.step()

        return loss_critic.item(), loss_actor.item(), rewards.sum()

    ''' helper function to obtain the Q-val of the next state'''

    def get_next_Val(self, next_states, actions, dones):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''

        ''' initialize zero value vectors '''
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)

        ''' update non-terminal state with Q value '''
        with torch.no_grad():
            for idx in range(batch_size):
                if not dones[idx]:
                    # Determine from which critic shall we extract the next value estimation
                    next_node = actions[idx]
                    # Get the next node state estimation
                    values[0, idx] = self.critics[next_node].network(next_states[idx].unsqueeze(0).float())
        return values


    def get_current_Val_and_policies(self, states, sources):
        batch_size = states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)
        policies = torch.zeros(batch_size, self.numNodes).to(self.device)
        idx = 0
        for src, state in zip(sources, states):
            values[0, idx] = self.critics[src].network(state.unsqueeze(0).float())
            policies[idx] = self.actors[src].policy_net(state.unsqueeze(0).float())
            idx += 1
        return values.squeeze(), policies


    ''' helper function to save the agents weights '''
    def save_agent(self, path):
        for idx in range(self.numAgents):
            torch.save(self.critics[idx].network.state_dict(), path + f'data/agent_{idx}_critic')
            torch.save(self.actors[idx].policy_net.state_dict(), path + f'data/agent_{idx}_policy')

    ''' helper function to load the agents weights '''
    def load_agent(self, path):
        for idx in range(self.numAgents):
            self.critics[idx].network.load_state_dict(torch.load(path + f'data/agent_{idx}_critic'))
            self.actors[idx].policy_net.load_state_dict(torch.load(path + f'data/agent_{idx}_policy'))
            self.actors[idx].policy_net.eval()
            self.critics[idx].network.eval()

class DecentralizedRelationalA2CAgent(DecentralizedA2CAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.numNodes = dynetwork.numBS
        self.actors = []
        self.critics = []
        self.init_neural_networks(setting)

    def init_neural_networks(self, setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        for i in range(self.config["nodes"]):
            actor = RelationalDQN(self.state_dim, setting=setting, device=self.device)
            critic = ValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        return

    def act(self,agentIdx, state, neighbor, dest):
        if dest in neighbor:
            return dest
        next_step = super().act(agentIdx, state, neighbor, dest)
        return next_step

class RelationalA2CAgent(CentralizedA2CAgent):
    """

            Initialize an instance of the agent class
            learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
            epsilon: probability that packets move randomly, instead of referencing routing policy
            discount: Degree to which we wish to maximize future rewards, value between (0,1)
            decay_rate: decays epsilon
            update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
            self.q: stores q-values
            *_memory: different methods of sampling from our memory bank

        """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.numNodes = self.config["nodes"]

    ''' helper function to generate the action-value function approximation for each agent '''

    def init_neural_networks(self, setting):
        ''' Initialize all neural networks with one neural network initialized for each node in the network. '''
        self.actor = RelationalDQN(self.state_dim, setting=setting, device=self.device)
        self.critic = RecurrentValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
        return

    """
    act is a function that gives a packet the next best action (if possible)
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank
    """


    def act(self, state, neighbor, dest):
        if dest in neighbor:
            return dest
        next_step = super().act(state, neighbor, dest)
        return next_step

    '''
        Updates replay memory with current experience
        and takes sample of previous experience
    '''

    def learn(self, transitions):
        ''' skip if no valid action or no reward is provided '''
        ''' check if our memory bank has sufficient memories to sample from '''
        if not transitions:
            return 0, 0, None

        states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions)

        '''extract values from experiences'''
        current_values = self.get_current_Val(states).squeeze()
        next_values = self.get_next_Val(next_states).squeeze()
        next_values = (~dones) * (next_values * self.config['gamma']) + rewards

        policy_pred = self.actor.policy_net(states)
        masks = [[1 if n in neighbor else 0 for n in range(self.numNodes)] for neighbor, dest in zip(neighbors, destinations)]
        numStates = states.shape[0]

        distributions = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
        neg_log_prob = -torch.log(distributions[range(numStates), actions])
        loss_actor = (neg_log_prob * (next_values - current_values)).mean()
        loss_critic = F.mse_loss(current_values, next_values)
        loss = loss_critic + loss_actor
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        '''backpropagation to update neural network'''
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.critic.network.parameters(), 10)
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return loss_critic.item(), loss_actor.item(), rewards.sum()

    ''' helper function to extract values from our stored experiences'''

    def extract_tensors(self, experiences):
        states = torch.cat(tuple(exps[0] for exps in experiences)).to(self.device)
        actions = torch.cat(tuple(torch.tensor([exps[1]]) for exps in experiences)).to(self.device)
        rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in experiences)).to(self.device)
        next_states = torch.cat(tuple(exps[3] for exps in experiences)).to(self.device)
        dones = torch.cat(tuple(torch.tensor([exps[4]]) for exps in experiences)).to(self.device)
        neighbors = tuple(exps[5] for exps in experiences)
        destinations = torch.cat(tuple(torch.tensor([exps[6]]) for exps in experiences)).to(self.device)
        return (states, actions, next_states, rewards, dones, neighbors, destinations)

    ''' helper function to obtain the Q-val of current state'''

    def get_current_Val(self, states):
        return self.critic.network(states.float())

    ''' helper function to obtain the Q-val of the next state'''

    def get_next_Val(self, next_states):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''
        return self.critic.network(next_states.float()).detach()

    ''' helper function to save the agents weights '''

    def save_agent(self, path):
        torch.save(self.critic.network.state_dict(), path + f'data/agent_critic')
        torch.save(self.actor.policy_net.state_dict(), path + f'data/agent_policy')

    ''' helper function to load the agents weights '''

    def load_agent(self, path):
        self.actor.policy_net.load_state_dict(torch.load(path + f'data/agent_policy'))
        self.actor.policy_net.eval()
        self.critic.network.load_state_dict(torch.load(path + f'data/agent_critic'))
        self.critic.network.eval()

class DecentralizedRelationalRecurrentA2CAgent(DecentralizedRelationalA2CAgent):
    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)

    ''' helper function to generate the action-value function approximation for each agent '''
    def init_neural_networks(self, setting):
        ''' Initialize all neural networks with one neural network initialized for each node in the network. '''
        for i in range(self.config["nodes"]):
            actor = RelationalRecurrentDQN(self.state_dim, setting=setting, device=self.device)
            critic = RecurrentValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        return

class RelationalRecurrentA2CAgent(RelationalA2CAgent):
    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)

    ''' helper function to generate the action-value function approximation for each agent '''
    def init_neural_networks(self, setting):
        ''' Initialize all neural networks with one neural network initialized for each node in the network. '''
        self.actor = LongLengthRelationalRecurrentDQN(self.state_dim, setting=setting, device=self.device)
        self.critic = LongLengthRecurrentValueDeepNetwork(self.state_dim, setting=setting, device=self.device)

    def learn(self, transitions):
        ''' skip if no valid action or no reward is provided '''
        ''' check if our memory bank has sufficient memories to sample from '''
        if not transitions:
            return 0, 0, None

        for idx in range(self.numNodes):
            if not transitions[idx]:
                continue
            states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions[idx])

            '''extract values from experiences'''
            current_values = self.get_current_Val(states.unsqueeze(1)).squeeze()
            next_values = self.get_next_Val(next_states.unsqueeze(1)).squeeze()
            next_values = (~dones) * (next_values * self.config['gamma']) + rewards

            policy_pred = self.actor.policy_net(states.unsqueeze(1))
            masks = [[1 if n in neighbor else 0 for n in range(self.numNodes)] for neighbor, dest in zip(neighbors, destinations)]
            numStates = states.shape[0]

            distributions = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
            neg_log_prob = -torch.log(distributions[range(numStates), actions])

            loss_actor = (neg_log_prob * (next_values - current_values)).mean()
            loss_critic = F.mse_loss(current_values, next_values)
            if idx == 0:
                losses_critic = loss_critic
                losses_actor = loss_actor
            else:
                losses_actor += loss_actor
                losses_critic += loss_critic
        loss = losses_critic + losses_actor
        self.critic.optimizer.zero_grad()
        self.actor.optimizer.zero_grad()
        '''backpropagation to update neural network'''
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.policy_net.parameters(), 10)
        torch.nn.utils.clip_grad_norm_(self.critic.network.parameters(), 10)
        self.actor.optimizer.step()
        self.critic.optimizer.step()
        return losses_critic.item(), losses_actor.item(), rewards.sum()

class DecentralizedRelationalRecurrentA2CAgentLongLength(DecentralizedRelationalA2CAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.numNodes = dynetwork.numBS
        self.actors = []
        self.critics = []
        self.init_neural_networks(setting)

    def init_neural_networks(self, setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        for i in range(self.config["nodes"]):
            actor = LongLengthRelationalRecurrentDQN(self.state_dim, setting=setting, device=self.device)
            critic = LongLengthRecurrentValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        return

    def act(self, agentIdx, state, neighbor, dest):
        if dest in neighbor:
            return dest
        next_step = super().act(agentIdx, [state], neighbor, dest)
        return next_step

    def get_current_Val_and_policies(self, states, idx):
        values = self.critics[idx].network(states.unsqueeze(0).float())
        policies = self.actors[idx].policy_net(states.unsqueeze(0).float())

        return values.squeeze(), policies.squeeze(0)


    def learn(self, transitions):
        if not transitions:
            return 0, 0, None

        for idx in range(self.numNodes):
            if not transitions[idx]:
                continue
            states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions[idx])
            ''' Extract Critics and Policy Estimation from Experience '''
            current_values, policy_pred = self.get_current_Val_and_policies(states, idx)
            ''' Get the Next State estimation and calculate the return to go estimation '''
            next_values = self.get_next_Val(next_states.unsqueeze(1), actions, dones).squeeze()
            next_values = (~dones) * (next_values * self.config['gamma']) + rewards
            ''' Generate Masks according to the current neighborhood '''
            masks = [[1 if ((n in neighbor) and (n < self.config["nodes"] or n == dest)) else 0 for n in range(self.numNodes)] for neighbor, dest in zip(neighbors, destinations)]
            numStates = states.shape[0]
            distributions = torch.cat([self.masked_softmax(policy_pred[p_idx], torch.tensor(masks[p_idx]), dim=-1).unsqueeze(0) for p_idx in range(numStates)])
            neg_log_prob = -torch.log(distributions[range(numStates), actions])
            loss_actor = (neg_log_prob * (next_values - current_values)).mean()
            loss_critic = F.mse_loss(current_values, next_values)

            if idx == 0:
                losses_critic = loss_critic
                losses_actor = loss_actor
            else:
                try:
                    losses_actor += loss_actor
                except:
                    losses_actor = loss_actor
                try:
                    losses_critic += loss_critic
                except:
                    losses_critic = loss_critic
        try:
            loss = losses_critic + losses_actor
            for agentIdx in range(self.numAgents):
                self.critics[agentIdx].optimizer.zero_grad()
                self.actors[agentIdx].optimizer.zero_grad()
            '''backpropagation to update neural network'''
            loss.backward()
            for agentIdx in range(self.numAgents):
                torch.nn.utils.clip_grad_norm_(self.actors[agentIdx].policy_net.parameters(), 10)
                torch.nn.utils.clip_grad_norm_(self.critics[agentIdx].network.parameters(), 10)
                self.actors[agentIdx].optimizer.step()
                self.critics[agentIdx].optimizer.step()
        except:
            return 0, 0, 0
        return losses_critic.item(), losses_actor.item(), rewards.sum()

class FedratedA2CAgent(DecentralizedRelationalA2CAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.main_actor = None
        self.main_critic = None
        self.init_main_neural_networks(setting)
        self.numAgents = len(self.actors)
        self.numUpdatesPerAgent = torch.zeros((self.numAgents))

    def init_main_neural_networks(self,setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        self.main_actor = RelationalDQN( self.state_dim, setting=setting, device=self.device)
        self.main_critic = ValueDeepNetwork(self.state_dim, setting=setting, device=self.device)

    ''' helper function to obtain the Q-val of current state'''
    def learn(self, transitions):
        if not transitions:
            return 0, 0, None

        states, actions, next_states, rewards, dones, neighbors, destinations = self.extract_tensors(transitions)
        sources = torch.cat(tuple(torch.tensor([exps[7]]) for exps in transitions)).to(self.device)
        ''' Extract Critics and Policy Estimation from Experience '''
        current_values, policy_pred = self.get_current_Val_and_policies(states, sources)
        ''' Get the Next State estimation and calculate the return to go estimation '''
        next_values = self.get_next_Val(next_states, actions, dones).squeeze()
        next_values = (~dones) * (next_values * self.config['gamma']) + rewards
        ''' Generate Masks according to the current neighborhood '''
        masks = [[1 if ((n in neighbor) and (n < self.config["nodes"] or n == dest)) else 0 for n in range(self.numNodes)] for neighbor, dest in
                 zip(neighbors, destinations)]
        numStates = states.shape[0]
        distributions = torch.cat([self.masked_softmax(policy_pred[idx], torch.tensor(masks[idx]), dim=-1).unsqueeze(0) for idx in range(numStates)])
        neg_log_prob = -torch.log(distributions[range(numStates), actions])
        loss_actor = (neg_log_prob * (next_values - current_values)).mean()
        loss_critic = F.mse_loss(current_values, next_values)
        loss = loss_critic + loss_actor
        for agentIdx in range(self.numAgents):
            self.critics[agentIdx].optimizer.zero_grad()
            self.actors[agentIdx].optimizer.zero_grad()
        '''backpropagation to update neural network'''
        loss.backward()
        for agentIdx in range(self.numAgents):
            torch.nn.utils.clip_grad_norm_(self.actors[agentIdx].policy_net.parameters(), 10)
            torch.nn.utils.clip_grad_norm_(self.critics[agentIdx].network.parameters(), 10)
            self.actors[agentIdx].optimizer.step()
            self.critics[agentIdx].optimizer.step()

        return loss_critic.item(), loss_actor.item(), rewards.sum()

    def update_central_controller_weights(self):
        totalUpdates = torch.sum(self.numUpdatesPerAgent)
        if totalUpdates != 0:
            weights = self.numUpdatesPerAgent / totalUpdates
            actor_avg = copy.deepcopy(self.actors[0].policy_net.state_dict())
            critic_avg = copy.deepcopy(self.critics[0].network.state_dict())
            # Actor Update
            for key in actor_avg.keys():
                actor_avg[key] = actor_avg[key] * weights[0]
                for idx in range(1, self.numAgents):
                    actor_avg[key] += self.actors[idx].policy_net.state_dict()[key] * weights[idx]
            # Critic Update
            for key in critic_avg.keys():
                critic_avg[key] = critic_avg[key] * weights[0]
                for idx in range(1, self.numAgents):
                    critic_avg[key] += self.critics[idx].network.state_dict()[key] * weights[idx]

            self.main_actor.policy_net.load_state_dict(actor_avg)
            self.main_critic.network.load_state_dict(critic_avg)
        # for idx in index:
        #     self.actors[idx].policy_net.load_state_dict(self.main_actor.policy_net.state_dict())
        #     self.critics[idx].network.load_state_dict(self.main_critic.network.state_dict())

        # Update weights at the distributed devices
        # for agent in range(self.numAgents):
        #     self.actors[agent].policy_net.load_state_dict(self.main_actor.policy_net.state_dict())
        #     self.critics[agent].network.load_state_dict(self.main_critic.network.state_dict())
        return


    def get_current_Val_and_policies(self, states, sources):
        batch_size = states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)
        policies = torch.zeros(batch_size, self.numNodes).to(self.device)
        idx = 0
        for src, state in zip(sources, states):
            values[0, idx] = self.critics[src].network(state.unsqueeze(0).float())
            policies[idx] = self.actors[src].policy_net(state.unsqueeze(0).float())
            self.numUpdatesPerAgent[src] += 1
            idx += 1
        return values.squeeze(), policies

class FedratedRelationalRecurrentA2CAgent(FedratedA2CAgent):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        super().__init__(dynetwork, setting, state_dim, device)
        self.main_actor = None
        self.main_critic = None
        self.init_main_neural_networks(setting)
        self.numAgents = len(self.actors)
        self.numUpdatesPerAgent = torch.zeros((self.numAgents))

    def init_neural_networks(self, setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        self.actors = []
        self.critics = []
        for i in range(self.config["nodes"]):
            actor = RelationalRecurrentDQN(self.state_dim, setting=setting, device=self.device)
            critic = RecurrentValueDeepNetwork(self.state_dim, setting=setting, device=self.device)
            self.actors.append(actor)
            self.critics.append(critic)
        return

    def init_main_neural_networks(self,setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        self.main_actor = RelationalRecurrentDQN( self.state_dim, setting=setting, device=self.device)
        self.main_critic = RecurrentValueDeepNetwork(self.state_dim, setting=setting, device=self.device)

class DQNAgent(object):
    """

        Initialize an instance of the agent class
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def __init__(self, dynetwork, setting, state_dim, device):
        self.config = {
            "nodes": dynetwork.numBS,
            "epsilon": setting['AGENT']['epsilon'],
            "decay_rate": setting['AGENT']['decay_epsilon_rate'],
            "batch_size": setting['DQN']['memory_batch_size'],
            "gamma": setting['AGENT']['gamma_for_next_q_val'],

            "update_less": setting['DQN']['optimize_per_episode'],
            "sample_memory": setting['AGENT']['use_random_sample_memory'],
            "recent_memory": setting['AGENT']['use_most_recent_memory'],
            "priority_memory": setting['AGENT']['use_priority_memory'],

            "update_epsilon": False,
            "update_models": torch.zeros([1, dynetwork.numBS], dtype=torch.bool),
            "entered": 0,
        }
        self.adjacency = dynetwork.adjacency_matrix
        self.state_dim = state_dim
        self.dqn = []
        self.device = device
        self.init_neural_networks(setting)


    ''' helper function to generate the action-value function approximation for each agent '''

    def init_neural_networks(self,setting):
        '''Initialize all neural networks with one neural network initialized for each node in the network.'''
        for i in range(self.config["nodes"]):
            temp_dqn = DQN(i, self.state_dim, setting=setting, device=self.device)
            self.dqn.append(temp_dqn)
        return

    '''Update the target neural network to match the policy neural network'''

    def update_target_weights(self):
        for nn in self.dqn:
            nn.target_net.load_state_dict(nn.policy_net.state_dict())

    def update_epsilon(self):
        self.config['epsilon'] = self.config["decay_rate"] * self.config['epsilon']
    """

    act is a function that gives a packet the next best action (if possible)
        learning rate: The amount of information that we wish to update our equation with, should be within (0,1]
        epsilon: probability that packets move randomly, instead of referencing routing policy
        discount: Degree to which we wish to maximize future rewards, value between (0,1)
        decay_rate: decays epsilon
        update_epsilon: utilized in our_env.router, only allows epsilon to decay once per time-step
        self.q: stores q-values
        *_memory: different methods of sampling from our memory bank

    """

    def act(self, agentIdx, state, neighbor, dest):
        ''' We will either random walk or reference Deep Q function approximation with probability epsilon '''
        available_guess = [n for n in neighbor if (n < self.config["nodes"] or (n-self.config["nodes"]) == dest)]
        if random.uniform(0, 1) < self.config['epsilon']:
            if not bool(available_guess):
                # checks if the packet's current node has any available neighbors
                return None
            else:
                # Explore action space
                next_step = random.choice(available_guess)
        else:
            if not bool(neighbor):
                return None
            else:
                ''' obtains the next best neighbor to move the packet from its current node by referencing our neural network '''
                with torch.no_grad():
                    state = state.to(self.device)
                    qvals = self.dqn[agentIdx].policy_net(state.float()).cpu()
                    next_step_idx = qvals[:, available_guess].argmax().item()
                    next_step = available_guess[next_step_idx]
        return next_step

    '''
        Updates replay memory with current experience
        and takes sample of previous experience
    '''
    def learn(self, dest, current_event, action, reward, next_state):
        ''' skip if no valid action or no reward is provided '''
        if (action == None) or (reward == None):
            pass
        else:
            nn = self.dqn[dest]
            if current_event != None:
                nn.replay_memory.push(current_event, action, next_state, reward)

            ''' check if our memory bank has sufficient memories to sample from '''
            if nn.replay_memory.can_provide_sample(self.config['batch_size']):

                '''check which type of memories to pull'''
                if self.config['sample_memory']:
                    experiences = nn.replay_memory.sample(self.config['batch_size'])

                elif self.config['recent_memory']:
                    experiences = nn.replay_memory.take_recent(self.config['batch_size'])

                elif self.config['priority_memory']:
                    experiences, experiences_idx = nn.replay_memory.take_priority(self.config['batch_size'])
                else:
                    raise Exception('Invalid Memory usage')

                states, actions, next_states, rewards, dones = self.extract_tensors(experiences)

                '''extract values from experiences'''
                current_q_values = self.get_current_QVal(nn.policy_net, states, actions)

                next_q_values = self.get_next_QVal(nn.policy_net, nn.target_net, next_states, actions, dones)

                target_q_values = (next_q_values * self.config['gamma']) + rewards

                '''update priority memory's probability'''
                if self.config['priority_memory']:
                    nn.replay_memory.update_priorities(experiences_idx, current_q_values, torch.transpose(target_q_values, 0, 1))

                '''backpropagation to update neural network'''
                loss = F.mse_loss(current_q_values, torch.transpose(target_q_values, 0, 1))
                nn.optimizer.zero_grad()
                loss.backward()
                nn.optimizer.step()
                return torch.mean(current_q_values - target_q_values).item()

    ''' helper function to extract values from our stored experiences'''

    def extract_tensors(self, experiences):
        states = torch.cat(tuple(exps[0] for exps in experiences)).to(self.device)
        actions = torch.cat(tuple(torch.tensor([exps[1]]) for exps in experiences)).to(self.device)
        next_states = torch.cat(tuple(exps[2] for exps in experiences)).to(self.device)
        rewards = torch.cat(tuple(torch.tensor([exps[3]]) for exps in experiences)).to(self.device)
        dones = torch.cat(tuple(torch.tensor([exps[4]]) for exps in experiences)).to(self.device)
        return (states, actions, next_states, rewards, dones)

    ''' helper function to obtain the Q-val of current state'''

    def get_current_QVal(self, policy_net, states, actions):
        return policy_net(states.float()).gather(dim=1, index=actions.unsqueeze(-1))

    ''' helper function to obtain the Q-val of the next state'''

    def get_next_QVal(self, policy_net, target_net, next_states, actions, dones):
        ''' need to apply neighbors mask to target_net() prior to taking the maximum '''
        target_q_val_est = target_net(next_states.float())
        curr_q_val_est = policy_net(next_states.float())

        ''' initialize zero value vectors '''
        batch_size = next_states.shape[0]
        values = torch.zeros(batch_size).view(1, -1).to(self.device)

        ''' update non-terminal state with Q value '''
        for idx in range(values.size()[1]):
            if not dones[idx]:
                # Extract the next node neighbors from the adjacency matrix
                adjs = self.adjacency[actions[idx]][0]
                adjs = (adjs == 1)
                # Extract the next node target Q values
                next_node_target_values = target_q_val_est[idx, :].view(1, -1)
                next_node_curr_values = curr_q_val_est[idx, :].view(1, -1)
                # Follow the greedy policy only from the next node available neighbors using double DQN method
                action_idx = torch.argmax(next_node_curr_values[adjs])
                values[0, idx] = next_node_target_values[adjs][action_idx].detach()

        return values

    ''' helper function to save the agents weights '''

    def save_agent(self, path):
        for nn in self.dqn:
            torch.save(nn.target_net.state_dict(), path+f'data/dest_{nn.ID}')

    ''' helper function to load the agents weights '''

    def load_agent(self, path):
        for nn in self.dqn:
            nn.target_net.load_state_dict(torch.load(path+f'data/dest_{nn.ID}'))
            nn.target_net.eval()
            nn.policy_net.load_state_dict(torch.load(path + f'data/dest_{nn.ID}'))
            nn.policy_net.eval()


