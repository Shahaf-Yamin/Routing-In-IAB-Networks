import numpy as np
from matplotlib import pyplot as plt
import networkx as nx
from copy import deepcopy
'''
Constants
'''
Boltzman_constant = 1.381e-23
temp_kelvin = 290
# 15 db
Noise_figure = 10 ** (15/10)

'''
Functions
'''
def calculate_noise_power(bandwidth):
    global temp_kelvin,Boltzman_constant,Noise_figure
    N_o = Noise_figure * Boltzman_constant * temp_kelvin * bandwidth # No = NF*K*T*B
    return 10*np.log10(N_o)

def calculate_path_loss(distance, frequency, tx_antenna_gain_db, rx_antenna_gain_db):
    '''
    :param distance: the physical distance
    :param frequency: the working frequency that we are trying to transmit in
    :param tx_antenna_gain_db: the TX antenna gain in decibels
    :param rx_antenna_gain_db: the RX antenna gain in decibels
    :return: Free space path loss in decibels
    '''
    FSPL_db = 20*np.log10(distance) + 20*np.log10(frequency) - 147.55 - tx_antenna_gain_db - rx_antenna_gain_db
    return FSPL_db

'''
Classes
'''
class NetworkNode(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.IAB_Fathers = []
        self.queueLength = 0
        self.relayCounter = 0
        self.dropCounter = 0


    def __eq__(self, other) -> bool:
        if self.x == other.x and self.y == other.y:
            return True
        else:
            return False

    def __ne__(self, other) -> bool:
        if self.x != other.x or self.y != other.y:
            return True
        else:
            return False

    def __sub__(self, other):
        return np.linalg.norm(np.array(self)-np.array(other), ord=2)

    def __str__(self) -> str:
        return f'({str(self.x)},{self.y})'

    def __array__(self) -> np.ndarray:
        return np.array([self.x, self.y])

class IABNetworkNode(NetworkNode):
    def __init__(self,x,y, TXpower):
        super(IABNetworkNode, self).__init__(x,y)
        self.IAB_Sons = []
        self.Users = []
        self.TXPower = TXpower

class UENetworkNode(NetworkNode):
    def __init__(self, x,y):
        super(UENetworkNode, self).__init__(x, y)

class NetworkTopologyEnv:
    def __init__(self, config):
        self.config = config
        self.scenario = config["NETWORK"]["Environment"]
        self.IAB = []
        self.UE = []
        self.donor = None
        self.FsplDb = None
        self.SNRMatrixDb = None
        self.spectralEff = None
        # Allocate Bs over grid
        self._allocate_bs_on_grid()
        # Allocate UE over grid
        self._allocate_ue_on_grid()
        # Calculate Path Loss Matrix
        self._calculate_path_loss_mat()
        self._calculate_SNR_mat()
        self._calculate_spectral_eff_mat()
        self._adjacenyMatrix = np.zeros_like(self.spectralEff)
        self.avaliableParents = config["IAB_Config"]["Parents"]
        self.avaliableUeParents = config["IAB_Config"]["Max Users Parents"]
        self.avaliableChildrens = config["IAB_Config"]["Childrens"]
        self.avaliableUEChildrens = config["IAB_Config"]["Max Users"]
        # An histogram that represents how many IAB father each BS/UE has
        self.numParents = np.zeros((self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']))
        # The donor cannot be a node children and therefore we subtract a single element
        self.numParents[0] = self.avaliableParents
        # An histogram that represents how many IAB childrens each BS has
        self.numChildrens = np.zeros((self.config.NETWORK['number Basestation']))
        self.edges = []

    def _allocate_bs_on_grid(self):
        '''
        Allocate all the BS over the Grid
        '''
        # Randomize locations
        x_coords = np.random.choice(self.config.NETWORK['max_x'], size=self.config.NETWORK['number Basestation'], replace=False)
        y_coords = np.random.choice(self.config.NETWORK['max_y'], size=self.config.NETWORK['number Basestation'], replace=False)

        # Generate node temporal classes
        self.donor = IABNetworkNode(0, 0, 25)
        self.IAB.append(self.donor)
        for cnt, iab_index in enumerate(range(self.config.NETWORK['number Basestation'] - 1)):
            node = IABNetworkNode(x_coords[cnt], y_coords[cnt], 15)
            self.IAB.append(node)

    def _allocate_ue_on_grid(self):
        '''
        Allocate all the BS over the Grid
        '''
        # Randomize locations
        x_coords = np.random.choice(self.config.NETWORK['max_x'], size=self.config.NETWORK['number user'], replace=False)
        y_coords = np.random.choice(self.config.NETWORK['max_y'], size=self.config.NETWORK['number user'], replace=False)

        for cnt, iab_index in enumerate(range(self.config.NETWORK['number user'] )):
            node = UENetworkNode(x_coords[cnt], y_coords[cnt])
            self.UE.append(node)

    def _calculate_path_loss_mat(self):
        '''
        Calculate free space path loss matrix
        '''
        self.FsplDb = np.zeros((self.config.NETWORK['number Basestation'], self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']))
        for tx_node in range(self.config.NETWORK['number Basestation']):
            for rx_node in range(self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']):
                tx = self.IAB[tx_node]
                rx = self.IAB[rx_node] if rx_node < self.config.NETWORK['number Basestation'] else self.UE[rx_node - self.config.NETWORK['number Basestation']]
                if rx == tx:
                    self.FsplDb[tx_node, rx_node] = np.inf
                    continue
                distance = tx - rx
                self.FsplDb[tx_node, rx_node] = calculate_path_loss(distance=distance, frequency=28 * (10 ** 9), tx_antenna_gain_db=10, rx_antenna_gain_db=0)

    def _calculate_SNR_mat(self):
        '''
        Calculate Signal To Noise Ratio assuming working under with a bandwidth of 400MHz
        '''
        self.SNRMatrixDb = np.zeros((self.config.NETWORK['number Basestation'], self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']))
        for tx_node in range(self.config.NETWORK['number Basestation']):
            for rx_node in range(self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']):
                tx = self.IAB[tx_node]
                #
                self.SNRMatrixDb[tx_node, rx_node] = tx.TXPower - self.FsplDb[tx_node, rx_node] - calculate_noise_power(bandwidth=400 * 10**6)

    def _calculate_spectral_eff_mat(self):
        self.spectralEff = np.log2(1+10**(self.SNRMatrixDb/20))

    def state(self):
        return self.spectralEff, self.numParents, self.numChildrens

    def step(self, txIdx, rxIdx):
        txChildrens = np.where(self._adjacenyMatrix[txIdx, :] != 0)[0].shape[0]
        rxParents = np.where(self._adjacenyMatrix[:, rxIdx] != 0)[0].shape[0]
        if rxIdx < len(self.IAB):
            self.IAB[rxIdx].IAB_Fathers.append(self.IAB[txIdx])
            self.IAB[txIdx].IAB_Sons.append(self.IAB[rxIdx])
            # self.edges.append((rxIdx, txIdx))
            if txChildrens == self.avaliableChildrens or rxParents == self.avaliableParents:
                raise Exception('Invalid action too many childrens or parents!')
        else:
            ue_relative_idx = rxIdx-self.config.NETWORK["number Basestation"]
            self.UE[ue_relative_idx].IAB_Fathers.append(self.IAB[txIdx])
            self.IAB[txIdx].Users.append(self.UE[ue_relative_idx])
            if txChildrens == (self.avaliableChildrens + self.avaliableUEChildrens) or rxParents == self.avaliableParents:
                raise Exception('Invalid action too many childrens or parents!')
            # if self.scenario != 'Shortest-Path':
            #     self.edges.append((rxIdx, txIdx))
        # Apply the action
        self._adjacenyMatrix[txIdx, rxIdx] = self.spectralEff[txIdx, rxIdx]
        self.numParents[rxIdx] += 1
        self.numChildrens[txIdx] += 1
        self.edges.append((txIdx, rxIdx))

        # Return the reward
        return self.spectralEff[txIdx, rxIdx]

    def calculateGraphScore(self):
        nodeScores = 10000*np.ones((self.config.NETWORK['number Basestation']), dtype=np.float64)
        for updateIdx in range(self.config.NETWORK['number Basestation']):
            # Update in iterative manner the overall node scores
            for node in range(self.config.NETWORK['number Basestation']):
                # Extract the Node parents indices
                parentsVec = np.argwhere(self._adjacenyMatrix[:, node] != 0)
                if len(parentsVec) != 0:
                    # Reset the current node score
                    nodeScores[node] = 0
                # Accumulate through all available node parents scores/spectral efficency values
                for parentNode in parentsVec:
                    nodeScores[node] += np.min((self._adjacenyMatrix[parentNode, node][0], nodeScores[parentNode][0]))
        return nodeScores[1:]

    def render(self):
        fig, ax = plt.subplots()
        x = self.nodes[0].x
        y = self.nodes[0].y
        donor = ax.scatter(x, y, s=200, zorder=1, color='r')
        ax.annotate('D', xy=(x, y), xytext=(5, 5), textcoords='offset points')

        x = []
        y = []
        iab_names = np.arange(0, len(self.nodes[1:]))
        for idx, iab in enumerate(self.nodes[1:]):
            x.append(iab.x)
            y.append(iab.y)

        x = np.array(x)
        y = np.array(y)
        size = [100 for elem in self.nodes[1:]]

        nodes = ax.scatter(x, y, s=np.array(size), zorder=1, color='b')
        for x, y, name in zip(x, y, iab_names):
            ax.annotate(str(name), xy=(x, y), xytext=(5, 5), textcoords='offset points')

        plt.legend((donor, nodes),
                   ('Donor', 'Node'),
                   scatterpoints=1,
                   loc='best',
                   ncol=2,
                   fontsize=8)

        plt.grid()
        for TX, RX in self.edges:
            x = np.array([self.nodes[TX].x, self.nodes[RX].x])
            y = np.array([self.nodes[TX].y, self.nodes[RX].y])
            if TX == 0 or RX == 0:
                backhaul = ax.plot(x, y, zorder=3, color='blue', linestyle='dotted')
            else:
                access = ax.plot(x, y, zorder=3, color='royalblue', linestyle='dashdot')

        plt.show()
        # plt.savefig(path + '/network_architecture_pyplot.png', dpi=1024)

class NetworkTopologyRLEnv(NetworkTopologyEnv):
    def __init__(self, config):
        super(NetworkTopologyRLEnv, self).__init__(config)
        # Generate a random congestion scenario.
        for iab in self.IAB:
            iab.queueLength = np.random.randint(1, 60)
            print(iab.queueLength)
        self.calculate_delay_matrix()
        self.generateNetworkGraph(config)
        self.NUM_BS = config.NETWORK["number Basestation"]
        self.NUM_UE = config.NETWORK["number user"]
        self.notAdjacenyMatrix = np.ones((self.NUM_BS, self.NUM_BS + self.NUM_UE)) - np.column_stack((np.eye(self.NUM_BS), np.zeros((self.NUM_BS, self.NUM_UE))))
        self.donor_mask = np.full(self.NUM_BS, False)
        self.donor_mask[0] = True
        self.initial_network = deepcopy(self.network)

    def reset(self):
        self.network = deepcopy(self.initial_network)
        # An histogram that represents how many IAB father each BS/UE has
        self.numParents = np.zeros((self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']))
        # The donor cannot be a node children and therefore we subtract a single element
        self.numParents[0] = self.avaliableParents
        # An histogram that represents how many IAB childrens each BS has
        self.numChildrens = np.zeros((self.config.NETWORK['number Basestation']))
        self.edges = []
        self.notAdjacenyMatrix = np.ones((self.NUM_BS, self.NUM_BS + self.NUM_UE)) - np.column_stack((np.eye(self.NUM_BS), np.zeros((self.NUM_BS, self.NUM_UE))))
        self._adjacenyMatrix = np.zeros_like(self.spectralEff)
        self.donor_mask = np.full(self.NUM_BS, False)
        self.donor_mask[0] = True
        for node in self.IAB:
            node.IAB_Sons = []
            node.Users = []
            node.IAB_Fathers = []
        for node in self.UE:
            node.IAB_Fathers = []

    def reset_topology(self):
        self.__init__(config=self.config)

    def generateNetworkGraph(self, config):
        self.network = nx.OrderedDiGraph()
        IABNodeIndices = [idx for idx in range(config.NETWORK['number Basestation'])]
        self.network.add_nodes_from(IABNodeIndices)
        UENodeIndices = [idx for idx in range(config.NETWORK['number Basestation'], config.NETWORK['number Basestation'] + config.NETWORK['number user'])]
        self.network.add_nodes_from(UENodeIndices)
        nx.set_edge_attributes(self.network, 0, 'edge_delay')

    def calculate_delay_matrix(self):
        self.delay = np.random.randint(2, 10, size=(self.SNRMatrixDb.shape[0], self.SNRMatrixDb.shape[1]))

    def state(self):
        # self.curr_adjacency = nx.convert_matrix.to_scipy_sparse_matrix(self.network, weight='edge_delay')
        # edge_index, edge_attr = convert.from_scipy_sparse_matrix(self.curr_adjacency)
        adjacencyMat = nx.adjacency_matrix(self.network).toarray()
        return self.calculateGraphDelayScore(), self.numParents, self.numChildrens, adjacencyMat

    def step(self, txIdx, rxIdx):
        prevScore = np.mean(self.calculateGraphDelayScore())
        _ = super(NetworkTopologyRLEnv, self).step(txIdx, rxIdx)
        self.network.add_edges_from([(txIdx, rxIdx)], edge_delay=self.delay[txIdx, rxIdx])
        nextState = self.state()
        reward = (prevScore - np.mean(nextState[0]))
        return nextState, reward, self.isDone()

    def isDone(self):
        #TODO: handle the user case over here
        possibleParents = np.argwhere(np.logical_and((self.numChildrens[:self.NUM_BS] < self.avaliableChildrens), np.logical_or(self.numParents[:self.NUM_BS] > 0, self.donor_mask))).ravel()
        NodesNumParents = self.numParents[self.numParents < self.avaliableParents].ravel()
        done = possibleParents.shape[0] == 0 or NodesNumParents.shape[0] == 0
        return done
        # possibleNodesForConnection = np.squeeze(np.argwhere(ParentsHist < availableParents))

    def calculateGraphDelayScore(self):
        nodeScores = np.zeros((self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']), dtype=np.float32)
        for updateIdx in range(self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']):
            # Update in iterative manner the overall node scores
            for node in range(self.config.NETWORK['number Basestation'] + self.config.NETWORK['number user']):
                # Extract the Node parents indices
                parentsVec = np.argwhere(self._adjacenyMatrix[:, node] != 0)
                numParents = len(parentsVec)
                # Reset the current node score to his queue length
                if node < self.config.NETWORK['number Basestation']:
                    nodeScores[node] = self.IAB[node].queueLength
                # Accumulate through all available node parents scores/spectral efficency values
                for parentNode in parentsVec:
                    nodeScores[node] += (self.delay[parentNode, node] + nodeScores[parentNode][0]) / numParents
        # nodeScores[0] = 0
        return np.expand_dims(nodeScores, axis=-1)