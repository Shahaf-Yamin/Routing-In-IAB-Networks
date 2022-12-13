import random
from . import Packet
import copy
import networkx as nx
import numpy as np

''' 
    Class created to store network and network attributes as well as generate packets 
    File contains functions:
        randomGeneratePackets: initialize packets to network in the beginning 
        GeneratePacket: generate additional packets as previous packets are delivered to keep network working in specified load
'''

class DynamicNetwork(object):
    def __init__(self, network, max_initializations=1000, packets=None, rejections=0, deliveries=0, load_arrival_rate = 0, numUsers=0):
        '''shared attributes'''
        self._network = copy.deepcopy(network)
        self._max_initializations = max_initializations
        self._packets = packets

        self.stats_exp_names = ["Average Delay Time", "Maximum Number of Packets a Single Node Hold", "Average Number of Packets a Single Node Hold",
                                "Percentile of Working Nodes at Maximal Queue Capacity", "Percentile of Empty Nodes",
                                "Maximal Normalized Throughput", "Maximal Link Utilization" ,"Average Transmission Rejections", "Arrival Ratio", "Fairness"]
        self.stats = {}
        for exp in self.stats:
            self.stats[exp] = []
        self.adjacency_matrix = nx.to_numpy_matrix(self._network, nodelist=sorted(list(self._network.nodes())))
        self._rejections = rejections
        self._deliveries = deliveries
        self.delayed_queue = []
        self._stripped_list = []
        self._delivery_times = []
        self._avg_link_utilization = []
        self._avg_router_fairness = []
        self._avg_throughput = []
        self._initializations = 0
        self._generations = 0
        self._max_queue_length = 0
        self._purgatory = []
        self._avg_q_len_arr = []
        self._num_empty_node=[]
        self._num_capacity_node = []
        self._num_working_node = []
        self._packet_drops = []
        self._packet_arrival = []
        self._hops_counter = []
        self._circles_counter = []
        self._free_packet_list = []
        self._lambda_load = load_arrival_rate
        self.numUsers = numUsers
        self.numBS = len(list(self._network.nodes())) - numUsers
        self.generated_packet_histogram = np.zeros((self.numBS))
        self.closed_stations = np.zeros((self.numBS), dtype=np.bool)

        if numUsers == 0:
            self.available_destinations = np.arange(self.numBS)
            self.generated_destination_histogram = np.zeros((self.numBS))
            self.destOffset = 0
        else:
            self.available_destinations = self.numBS + np.arange(self.numUsers)
            self.generated_destination_histogram = np.zeros((numUsers))
            self.destOffset = self.numBS
        self.load = 0
        self.set_directed_acyclic_relationship()
        self.optimalNumHops = dict(nx.shortest_path_length(self._network))

    def randomGeneratePackets(self, num_packets_to_generate):
        ''' Function used to generate packets handle both first initialization or later additional packet injections '''
        self._initializations = num_packets_to_generate
        self._generations = num_packets_to_generate
        tempList = {}
        notfull = list(range(self.numBS))
        for index in range(num_packets_to_generate):
            currPack, notfull = self.GeneratePacket(index=index, wait=0, midSim=False, notfull=copy.deepcopy(notfull))
            '''put curPack into startNode's queue'''
            self._network.nodes[currPack.get_startPos()]['sending_queue'].append(currPack)
            self.generated_packet_histogram[currPack.get_startPos()] += 1
            self.generated_destination_histogram[currPack.get_endPos() - self.destOffset] += 1
            tempList[index] = currPack
        '''create Packets Object'''
        packetsObj = Packet.Packets(tempList)

        '''Assign Packets Object to the network'''
        self._packets = copy.deepcopy(packetsObj)
        del packetsObj
        del tempList

    """ called by randomGeneratePackets when generating additional packets after previous packets are delivered """
    def GeneratePacket(self, index, wait=0, midSim = True, notfull=None, errorGeneration=False):
        """checks to see if we have exceed the maximum number of packets allocated in the simulation"""
        sending_queue = 'sending_queue'
        receiving_queue = 'receiving_queue'
        packets = self._packets
        if wait <= 0:
            """ creates a list of not full nodes to check during new packet creation """ 
            if midSim:
                notfull = list(range(self.numBS))
            startNode = random.choice(notfull)
            endNode = random.choice(self.pathLut[startNode])
            """ searches through notfull list until an available node is located for initial packet assignment """ 
            while (len(self._network.nodes[startNode][sending_queue]) + len(self._network.nodes[startNode][receiving_queue])
                   >= self._network.nodes[startNode]['max_receive_capacity']):
                notfull.remove(startNode)
                try:
                    startNode = random.choice(notfull)
                except:
                    print("Error: All Nodes are Full")
                    return
            """ searches through notfull list until an available node is located for initial packet assignment """ 
            
            """ assigns the packet different delivery destination than starting point """ 
            while (startNode == endNode):
                endNode = random.choice(self.pathLut[startNode])
                
            curPack = Packet.Packet(startNode, endNode, startNode, index, 0)
            self.generated_packet_histogram[startNode] += 1
            self.generated_destination_histogram[endNode - self.destOffset] += 1

            if midSim:
                """ appends newly generated packet to startNodes queue """ 
                packets.packetList[index] = curPack
                # Increase by 1 the number of middle simulation initializations
                self._initializations += 1
                # Add the generated packet to the node sending_queue queue with no wait indicators
                self._network.nodes[curPack.get_startPos()][sending_queue].append(curPack)
            else:
                ''' This case we update sequentially each node queue'''
                return curPack, notfull
        else:
            self._purgatory.append((index, wait - 1))
            raise Exception('Invalid Case')

        return

    def add_free_packet_idx(self, packet_idx):
        self._free_packet_list.append(packet_idx)

    def set_directed_acyclic_relationship(self):
        self.pathLut = {}
        self.numberOfAttachedUsers = []
        if self.numUsers != 0:
            for TX in range(self.numBS):
                self.pathLut[TX] = []
                numberOfAttachedUsers = 0
                for RX in range(self.numBS, self.numUsers+self.numBS):
                    if nx.has_path(self._network, TX, RX):
                        self.pathLut[TX].append(RX)
                        numberOfAttachedUsers += 1
                self.numberOfAttachedUsers.append(numberOfAttachedUsers)
            self.numberOfAttachedUsers = np.array(self.numberOfAttachedUsers)
            self.probGenerationNodes = self.numberOfAttachedUsers[1:] / self.numberOfAttachedUsers[1:].sum()
        else:
            for TX in range(self.numBS):
                numberOfAttachedBs = 0
                self.pathLut[TX] = []
                for RX in range(self.numBS):
                    if nx.has_path(self._network, TX, RX) and RX != TX:
                        self.pathLut[TX].append(RX)
                        numberOfAttachedBs += 1
                self.numberOfAttachedUsers.append(numberOfAttachedBs)
            self.numberOfAttachedUsers = np.array(self.numberOfAttachedUsers)
            self.probGenerationNodes = np.ones((self.numBS-1)) / (self.numBS-1)

    def getCurrTimeSlotPackets(self):
        """checks to see if we have exceed the maximum number of packets allocated in the simulation"""
        sending_queue = 'sending_queue'
        receiving_queue = 'receiving_queue'
        packets = self._packets
        ''' Randomize how many packets each node will receive at the next time slot by sampling from poisson distribution '''
        numPacketsVec = np.random.poisson(self._lambda_load-1, 1)
        ''' We remove 1 from lambda and insert zero to the chosen start nodes because we are inserting the IAB Donor every timeslot '''
        # startNodes = np.concatenate((np.random.choice(np.arange(0, self.numBS), size=numPacketsVec[0]), np.array([0])))
        DonorIdx = 0
        startNodes = [DonorIdx for idx in range(np.random.poisson(0.8 * self._network.nodes[DonorIdx]['max_send_capacity'], 1)[0])] + [np.random.choice(np.arange(1, self.numBS), p=self.probGenerationNodes) for idx in range(numPacketsVec[0])]

        """ creates a list of not full nodes to check during new packet creation """
        for startNode in startNodes:
            # for idx in range(numPacketsVec[startNode]):
            endNode = startNode
            '''
            Randomize the next destination uniformly 
            '''
            while endNode == startNode:
                endNode = random.choice(self.pathLut[startNode])

            ''' Generate the next packet '''
            if self._free_packet_list:
                ''' This case we have free packet to choose from the list '''
                index = self._free_packet_list.pop()
                curPack = Packet.Packet(startNode, endNode, startNode, index, 0, time=0, drops=packets.packetList[index].get_drops())
                packets.packetList[index] = curPack
            else:
                ''' This case we didn't have any free packet to choose from the available packet's list and therefore we add new packet to the system '''
                index = self._initializations # save the packet new index
                self._initializations += 1 # increment the packet initialization by 1
                curPack = Packet.Packet(startNode, endNode, startNode, index, 0)
                packets.append(curPack)
            self._generations += 1
            self.generated_packet_histogram[startNode] += 1
            self.generated_destination_histogram[endNode-self.destOffset] += 1
            # Add the generated packet to the node sending_queue queue with no wait indicators
            self._network.nodes[curPack.get_startPos()][sending_queue].append(curPack)
        return

