from . import dynetwork
from . import UpdateEdges as UE
import gym
import networkx as nx
import torch

import copy
import numpy as np
import math
import os
import random
from .Rewards import reward1, reward2, reward3, reward4, reward5, reward6, reward7, reward8
from .Buffer import  PriorityQueue
import matplotlib.pyplot as plt
from Utils import topology_desginer

""" 
This class contains our gym environment which contains all of the necessary components for agents to take actions and receive rewards. file contains functions: 
change_network: edge deletion/re-establish, edge weight change
purgatory: queue to generate additional queues as previous packets are delivered
step: obtain rewards for updating Q-table after an action
is_capacity: check if next node is full and unable to receive packets
send_packet: attempt to send packet to next node
reset: reset environment after each episode
resetForTest: reset environment for each trial (test for different network loads)
get_state: obtain packet's position info
update_queues: update each nodes packet holding queue
update_time: update packet delivery time
calc_avg_delivery: helper function to calculate delivery time
router: used to route all packets in ONE time stamp
updateWhole: helper function update network environment and packets status
"""

class dynetworkEnvBaseline(gym.Env):
    '''Initialization of the network'''
    def __init__(self, algorithm, setting):
        # Configuration related variables
        self.nnodes = setting["NETWORK"]["number Basestation"]
        self.num_user = setting["NETWORK"]["number user"]
        self.num_associated_iab_parents = setting["IAB_Config"]["Max Users Parents"]
        self.num_associated_users = setting["IAB_Config"]["Max Users"]
        self.max_queue = setting["NETWORK"]["holding capacity"]
        self.max_transmit = setting["NETWORK"]["sending capacity"]
        self.npackets = setting["NETWORK"]["initial num packets"]
        self.max_initializations = setting["NETWORK"]["max_additional_packets"]
        self.max_edge_weight = setting["NETWORK"]["max_edge_weight"]
        self.max_edge_per = setting["NETWORK"]["max_edge_per"]
        self.min_edge_removal = setting["NETWORK"]["min_edge_removal"]
        self.max_edge_removal = setting["NETWORK"]["max_edge_removal"]
        self.packet_maximal_latency_threshold = setting["NETWORK"]["MAX_TTL"]
        self.edge_change_type = setting["NETWORK"]["edge_change_type"]
        self.network_type = setting["NETWORK"]["network_type"]
        self.arrival_load = setting["NETWORK"]["lambda_arrival_rate"]
        # Network Simulation related variables
        self.current_timeslot = 0
        self.simulation_ending_time = None
        self.router_type = 'dijkstra'
        self.initial_dynetwork = None
        self.dynetwork = None
        self.print_edge_weights = True
        self.packet = -1
        self.curr_queue = PriorityQueue(self.max_queue, is_fifo=setting["NETWORK"]["is_fifo"])
        self.remaining = PriorityQueue(self.max_queue, is_fifo=setting["NETWORK"]["is_fifo"])
        self.nodes_traversed = 0
        # Statistics related variables
        self.relayed_packet_histogram = np.zeros((self.nnodes))
        stateSpace = self.num_user if self.num_user != 0 else self.nnodes
        self.arrived_packet_histogram = np.zeros((stateSpace))
        self.dropped_packet_histogram = np.zeros((self.nnodes))
        self.pathMapping = np.zeros(((self.nnodes, stateSpace)))
        self.dropMapping = np.zeros(((self.nnodes, stateSpace)))
        self.PolicyMapping = np.zeros(((stateSpace, self.nnodes, self.num_user + self.nnodes)))
        self.dest_offset = self.nnodes if self.num_user != 0 else 0
        self.dropped_packets_counter = 0
        self.arrived_packet_counter = 0
        self.circles_counter = 0
        self.hops_counter = []
        self.setting = setting
        self.initialize_network_graph()
        self.results_dir = setting.result_dir
        self.dynetwork = copy.deepcopy(self.initial_dynetwork)
        self.optimalNumHops = dict(nx.shortest_path_length(self.initial_dynetwork._network))
        self.updateDelayMapping()
        '''use dynetwork class method randomGeneratePackets to populate the network with packets'''
        self.dynetwork.randomGeneratePackets(copy.deepcopy(self.npackets))

        ''' Reward normalization variables '''
        self.max_reward = 0
        self.avg_reward = 0
        self.alpha = 0.01

        self.render()

        self.user_association_rule_one_hot = torch.zeros((self.num_user, self.nnodes))
        for nodeIdx in range(self.nnodes):
            neighbors_list = list(self.dynetwork._network.neighbors(nodeIdx))
            for dest in range(self.num_user):
                offseted_dest = dest + self.dest_offset
                if offseted_dest in neighbors_list:
                    self.user_association_rule_one_hot[dest][nodeIdx] = 1
        self.maxGrid = setting["NETWORK"]["max_y"]
        self.ue_speed = setting["NETWORK"]["ue_speed"]
        self.UsersMovementClass = UE.DirectedUserMovement(self.dynetwork, self.nnodes, self.num_user, self.num_associated_users, self.num_associated_iab_parents, self.maxGrid, speed=self.ue_speed)

        self.ttl_avg = np.zeros((self.nnodes))
        self.ttl_avg_square = np.zeros((self.nnodes))
        self.ttl_idx = np.zeros((self.nnodes),dtype=np.int)
        self.ttl_hist = np.infty * np.ones((self.nnodes, 1000))
        self.queue_avg_square = np.zeros((self.nnodes))
        self.queue_avg = np.zeros((self.nnodes))
        self.queue_idx = np.zeros((self.nnodes),dtype=np.int)
        self.queue_hist = np.infty * np.ones((self.nnodes, 1000))
        self.reward_avg_square = np.zeros((self.nnodes))
        self.reward_avg = np.zeros((self.nnodes))
        self.reward_idx = np.zeros((self.nnodes),dtype=np.int)
        self.reward_hist = np.infty * np.ones((self.nnodes, 1000))


    def updateDelayMapping(self):
        self.trip_delay_mapping = nx.floyd_warshall_numpy(self.dynetwork._network,self.dynetwork._network.nodes, weight='edge_global_delay')

    def initialize_network_graph(self):
        '''Initialize a dynetwork object using Networkx and dynetwork.py'''
        ''' Generate the current network Topology and add attributes to each node'''
        network, positions = topology_desginer.generateIABNetwork(config=self.setting)
        self._positions = positions

        '''Set Edge attributes'''
        nx.set_edge_attributes(network, 0, 'edge_delay')
        nx.set_edge_attributes(network, 0, 'edge_global_delay')
        nx.set_edge_attributes(network, 0, 'edge_error_probability')
        nx.set_edge_attributes(network, 0, 'sine_state')
        for s_edge, e_edge in network.edges:
            network[s_edge][e_edge]['edge_delay'] = random.randint(1, self.max_edge_weight)
            network[s_edge][e_edge]['edge_global_delay'] = network[s_edge][e_edge]['edge_delay']
            network[s_edge][e_edge]['initial_weight'] = network[s_edge][e_edge]['edge_delay']
            network[s_edge][e_edge]['edge_error_probability'] = random.uniform(0, self.max_edge_per)
            network[s_edge][e_edge]['initial_per'] = network[s_edge][e_edge]['edge_error_probability']
            network[s_edge][e_edge]['sine_state'] = random.uniform(0, math.pi)

        ''' make a copy so that we can preserve the initial state of the network '''
        self.initial_dynetwork = dynetwork.DynamicNetwork(copy.deepcopy(network), self.max_initializations, load_arrival_rate=self.setting["NETWORK"]["lambda_arrival_rate"], numUsers=self.setting["NETWORK"]['number user'])
        return

    '''helper function to save the statistics of the current timeslot'''
    def save_stats(self, max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, reward):
        '''helper function to save the statistics of the current timeslot'''

        '''Congestion Measure #1: max queue len'''
        self.dynetwork._max_queue_length = max_queue_length

        '''Congestion Measure #2: avg queue length pt2'''
        self.dynetwork._avg_q_len_arr.append(np.mean(node_queue_lengths))

        '''Congestion Measure #3: percent node at capacity'''
        self.dynetwork._num_capacity_node.append(num_nodes_at_capacity)

        '''Congestion Measure #4: percentile of nodes with queue of packets'''
        self.dynetwork._num_working_node.append(num_nonEmpty_nodes)

        '''Congestion Measure #5: percent empty nodes'''
        self.dynetwork._num_empty_node.append(self.nnodes - num_nonEmpty_nodes)

        """ Congestion Measure #6: Link Utilization Percentile """
        self.dynetwork._avg_link_utilization.append(np.sum(link_utilization['Links']) / np.sum(link_utilization['availableLinks']))

        """ Congestion Measure #7: Average Normalized Throughput """
        self.dynetwork._avg_throughput.append(np.mean(network_throughput))

        '''Congestion Measure #8: Fairness -> we measure the fairness among network routers by min(links)/max(links) '''
        try:
            normalized_arrival_rate = self.arrived_packet_histogram / (self.dynetwork.generated_destination_histogram + 1e-5)
            Fairness_criteria = np.min(normalized_arrival_rate) / np.max(normalized_arrival_rate)
        except ZeroDivisionError:
            Fairness_criteria = np.nan
        self.dynetwork._avg_router_fairness.append(Fairness_criteria)

        self.dynetwork._packet_drops.append(self.dropped_packets_counter)
        self.dropped_packets_counter = 0
        self.dynetwork._packet_arrival.append(self.arrived_packet_counter)
        self.arrived_packet_counter = 0

        self.dynetwork._circles_counter.append(self.circles_counter)
        self.circles_counter = 0
        if len(self.hops_counter) != 0:
            self.dynetwork._hops_counter.append(np.mean(self.hops_counter))
        self.hops_counter = []



        ''' Congestion Measure #9: Rejections '''
        self.dynetwork._rejections = rejections

        self.accumualted_reward = reward

    '''helper function to update learning environment in each time stamp'''

    '''Use to update edges in network'''
    def change_network(self):
        UE.Delete(self.dynetwork, self.min_edge_removal, self.max_edge_removal)
        UE.Restore(self.dynetwork, self.max_edge_removal)
        if self.edge_change_type == 'none':
            UE.updateGlobalEdges(self.dynetwork)
        elif self.edge_change_type == 'sinusoidal':
            UE.Sinusoidal(self.dynetwork)
        elif self.edge_change_type == 'sinusoidalGlobal':
            UE.SinusoidalGlobal(self.dynetwork)
        else:
            UE.Random_Walk(self.dynetwork)

    '''Method for emptying 'purgatory' which holds indices of packets that have
       been delivered so they may be reused'''

    '''Takes packets which are now ready to be sent and puts them in the sending queue of the node '''
    def update_queues(self):
        sending_queue = 'sending_queue'
        receiving_queue = 'receiving_queue'
        for nodeIdx in self.dynetwork._network.nodes:
            node = self.dynetwork._network.nodes[nodeIdx]
            node['growth'] = len(node[receiving_queue])
            queue = copy.deepcopy(node[receiving_queue])
            for elt in queue:
                '''increment packet delivery time step'''
                packetIdx, wait = elt
                if wait == 0:
                    ''' This case the delay induced by the link is over and we pass this packet to the sending queue '''
                    node[sending_queue].append(self.dynetwork._packets.packetList[packetIdx])
                    ''' We sign that this node successfully received a packet '''
                    self.relayed_packet_histogram[nodeIdx] += 1
                    ''' Remove this packet ID from the receiving queue '''
                    node[receiving_queue].remove(elt)
                else:
                    ''' This case the delay induced by the link is not over and the packet remains at the receiving queue '''
                    idx = node[receiving_queue].index(elt)
                    ''' Reduce by 1 the packet wait time '''
                    node[receiving_queue][idx] = (packetIdx, wait - 1)


    ''' Update time spent in queues for each packets '''
    def handle_packet_drop_event(self, packet_idx, node, nodeIdx):
        ''' The packet time elapsed the maximal amount of time and therfore, we drop the packet '''
        node['sending_queue'].remove(packet_idx)
        ''' Add this packet to the purgatory and reset her time '''
        self.dynetwork.add_free_packet_idx(packet_idx)
        packet = self.dynetwork._packets.packetList[packet_idx]
        steps_set = set()
        dest = packet.get_endPos()
        prevStep = None
        for item in packet._steps:
            step, time = item
            if step in steps_set:
                self.circles_counter += 1
            else:
                steps_set.add(step)
            self.dropMapping[step][dest-self.dest_offset] += 1
            if prevStep is not None:
                self.PolicyMapping[dest-self.dest_offset][prevStep][step] += 1
            prevStep = step


        '''
        Increase the drop counter of this packet
        '''
        packet.increase_drops()

        ''' Update the drop packet statistics '''
        self.dropped_packet_histogram[nodeIdx] += 1

        # Increase the drop counter by 1
        self.dropped_packets_counter += 1

    def update_time(self):
        sending_queue = 'sending_queue'
        self.current_timeslot += 1
        for nodeIdx, node in self.dynetwork._network.nodes.items():
            for packetInstance in node[sending_queue]:
                curr_time = packetInstance.get_time()
                if curr_time + 1 > self.packet_maximal_latency_threshold:
                    self.handle_packet_drop_event(packet_idx=packetInstance.get_index(), node=node, nodeIdx=nodeIdx)
                else:
                    ''' This case we only increase by 1 the packet time '''
                    packetInstance.set_time(curr_time + 1)
    def apply_movement(self):
        userChanges = self.UsersMovementClass.applyMovement(timeslot=self.current_timeslot, closedStations=self.dynetwork.closed_stations)
        # histogram = np.zeros((self.nnodes))
        for ue, changes in userChanges.items():
            self.user_association_rule_one_hot[ue-self.dest_offset][changes['disconnected-connections']] = 0
            self.user_association_rule_one_hot[ue-self.dest_offset][changes['new-connections']] = 1
            # histogram[changes['disconnected-connections']] += 1
            # histogram[changes['new-connections']] += 1


    def updateWholeBaseline(self):
        # self.render_pyplot(self.current_timeslot)
        # Change the topology structure due to the wireless medium
        self.change_network()
        # Apply Users Movement
        self.apply_movement()
        if self.current_timeslot < self.simulation_ending_time - self.packet_maximal_latency_threshold:
            # Inject more packets to the system only in case that we have some time until we finish our simulation
            self.dynetwork.getCurrTimeSlotPackets()
        self.update_queues()
        self.update_time()

    """ checks to see if there is space in target_nodes queue """

    def is_capacity(self, target_node):
        sending_queue = 'sending_queue'
        receiving_queue = 'receiving_queue'
        total_queue_len = len(self.dynetwork._network.nodes[target_node][sending_queue]) + \
                          len(self.dynetwork._network.nodes[target_node][receiving_queue])
        return total_queue_len >= self.dynetwork._network.nodes[target_node]['max_receive_capacity']

    """ this function resets the environment """

    def reset_env(self, rewardfun=None, seed=0, reset_stochastic_engine=False, arrival_rate=None, resetTopology=False):
        if reset_stochastic_engine:
            np.random.seed(seed)
            random.seed(seed)

        if resetTopology is True:
            self.initialize_network_graph()
        self.dynetwork = copy.deepcopy(self.initial_dynetwork)

        # Reset timeline variable
        self.current_timeslot = 0
        self.dropped_packets_counter = 0
        self.arrived_packet_counter = 0

        '''
        Reset Train statistics
        '''
        self.td_error_mean = []
        self.td_error_max = []
        self.td_error_min = []
        stateSpace = self.num_user if self.num_user != 0 else self.nnodes
        self.relayed_packet_histogram = np.zeros((self.nnodes))
        self.arrived_packet_histogram = np.zeros((stateSpace))
        self.dropped_packet_histogram = np.zeros((self.nnodes))
        self.pathMapping = np.zeros((self.nnodes, stateSpace))
        self.dropMapping = np.zeros((self.nnodes, stateSpace))
        self.PolicyMapping = np.zeros(((stateSpace, self.nnodes, self.num_user + self.nnodes)))
        self.dynetwork.randomGeneratePackets(self.npackets)
        if arrival_rate is None:
            arrival_rate = self.arrival_load
        else:
            self.arrival_load = arrival_rate
        self.dynetwork._lambda_load = arrival_rate
        self.updateDelayMapping()
        self.user_association_rule_one_hot = torch.zeros((self.num_user, self.nnodes))
        for nodeIdx in range(self.nnodes):
            neighbors_list = list(self.dynetwork._network.neighbors(nodeIdx))
            for dest in range(self.num_user):
                offseted_dest = dest + self.dest_offset
                if offseted_dest in neighbors_list:
                    self.user_association_rule_one_hot[dest][nodeIdx] = 1

        self.UsersMovementClass = UE.DirectedUserMovement(self.dynetwork, self.nnodes, self.num_user, self.num_associated_users, self.num_associated_iab_parents, self.maxGrid, speed=self.ue_speed)

        # self.ttl_avg = np.zeros((self.nnodes))
        # self.ttl_avg_square = np.zeros((self.nnodes))
        # self.ttl_idx = np.zeros((self.nnodes), dtype=np.int)
        # self.ttl_hist = np.infty * np.ones((self.nnodes, 100))
        # self.queue_avg_square = np.zeros((self.nnodes))
        # self.queue_avg = np.zeros((self.nnodes))
        # self.queue_idx = np.zeros((self.nnodes), dtype=np.int)
        # self.queue_hist = np.infty * np.ones((self.nnodes, 100))
        # self.reward_avg_square = np.zeros((self.nnodes))
        # self.reward_avg = np.zeros((self.nnodes))
        # self.reward_idx = np.zeros((self.nnodes), dtype=np.int)
        # self.reward_hist = np.infty * np.ones((self.nnodes, 100))

        print('Environment reset')

    '''helper function to calculate delivery times'''

    def calc_avg_delivery(self):
        delivery_times = self.dynetwork._delivery_times
        try:
            avg_delivery = (sum(delivery_times) / len(delivery_times))
        except ZeroDivisionError:
            avg_delivery = 0
        return avg_delivery

    ''' Save an image of the current state of the network'''
    def render_pyplot(self, timestep=0):
        fig, ax = plt.subplots()
        self._positions = nx.get_node_attributes(self.dynetwork._network, 'location')

        x, y = self._positions[0]
        donor = ax.scatter(x, y, s=200, zorder=1, color='r')
        ax.annotate('D', xy=(x, y), xytext=(5, 5), textcoords='offset points')

        '''
        IAB Nodes
        '''
        x_iab = []
        y_iab = []
        iab_names = np.arange(1, self.nnodes)
        for idx in range(1, self.nnodes):
            temp_x, temp_y = self._positions[idx]
            x_iab.append(temp_x)
            y_iab.append(temp_y)

        x_iab = np.array(x_iab)
        y_iab = np.array(y_iab)
        size = [100 for elem in range(self.nnodes-1)]

        # Plot a circle of the BS/Relay
        nodes = ax.scatter(x_iab, y_iab, s=np.array(size), zorder=1, color='b')
        for x, y, name in zip(x_iab, y_iab, iab_names):
            # Write the BS/Relay Name above the corresponding circle
            ax.annotate(str(name), xy=(x, y), xytext=(5, 5), textcoords='offset points')
        # Add the donor location
        x_iab = np.squeeze(np.hstack((np.array([[0]]), np.expand_dims(x_iab, axis=0))))
        y_iab = np.squeeze(np.hstack((np.array([[0]]), np.expand_dims(y_iab, axis=0))))
        '''
        User Equipment 
        '''
        x_user = []
        y_user = []
        ue_names = np.arange(0, self.num_user)
        for idx in range(self.nnodes, self.num_user+self.nnodes):
            temp_x, temp_y = self._positions[idx]
            x_user.append(temp_x)
            y_user.append(temp_y)

        x_user = np.array(x_user, dtype=np.int64)
        y_user = np.array(y_user, dtype=np.int64)
        size = [10 for elem in range(self.num_user)]

        # Plot a circle of the BS/Relay
        ues = ax.scatter(x_user, y_user, s=np.array(size), zorder=1, color='g')
        # for x, y, name in zip(x_user, y_user, ue_names):
        #     # Write the BS/Relay Name above the corresponding circle
        #     ax.annotate(str(name), xy=(x, y), xytext=(5, 5), textcoords='offset points')

        nodes_legned = plt.legend((donor, nodes, ues),
                                  ('Donor', 'Node', 'UE'),
                                  scatterpoints=1,
                                  loc='upper left',
                                  ncol=2,
                                  fontsize=8)

        plt.grid()
        NUM_BASESTATIONS = self.nnodes
        for TX_idx, RX_idx in self.dynetwork._network.edges:
            if TX_idx >= NUM_BASESTATIONS:
                continue
            TX_x, TX_y = x_iab[TX_idx], y_iab[TX_idx]
            if RX_idx >= NUM_BASESTATIONS:
                RX_x, RX_y = x_user[RX_idx - NUM_BASESTATIONS], y_user[RX_idx - NUM_BASESTATIONS]
                access = True
            else:
                RX_x, RX_y = x_iab[RX_idx], y_iab[RX_idx]
                access = False

            if access:
                x = np.array([TX_x, RX_x])
                y = np.array([TX_y, RX_y])
                access = ax.plot(x, y, zorder=1, color='royalblue', linestyle='dashdot', alpha=0.2, label='access')
            else:
                backhaul = ax.annotate("", xy=(RX_x, RX_y), xytext=(TX_x, TX_y), arrowprops=dict(arrowstyle="->"), color='black', label='backhaul')
                x = np.array([TX_x, RX_x])
                y = np.array([TX_y, RX_y])
                backhaul = ax.plot(x, y, zorder=3, color='black')

        lines = ax.get_lines()
        legend1 = plt.legend([lines[i] for i in [0, -1]], ["Backhaul", "Access"], loc=4)
        ax.add_artist(legend1)
        ax.add_artist(nodes_legned)
        plt.xlim([0, self.maxGrid])
        plt.ylim([0, self.maxGrid])
        plt.savefig(f"{self.results_dir}/network_topology_timestep_{timestep}.png", dpi=512)

    def render(self, timestep=0, episode=0, queue_mode = False):
        node_labels = {}
        plt.figure()
        if queue_mode:
            for node in self.dynetwork._network.nodes:
                node_labels[node] = len(self.dynetwork._network.nodes[node]['sending_queue']) + len(self.dynetwork._network.nodes[node]['receiving_queue'])
        else:
            for node in self.dynetwork._network.nodes:
                node_labels[node] = node
        self._positions = nx.get_node_attributes(self.dynetwork._network, 'location')
        nx.draw(self.dynetwork._network, pos=self._positions, labels=node_labels, font_weight='bold')
        # if self.print_edge_weights:
        #     edge_labels = nx.get_edge_attributes(self.dynetwork._network, 'edge_delay')
        #     nx.draw_networkx_edge_labels(self.dynetwork._network, pos=self._positions, edge_labels=edge_labels)


        plt.axis('off')
        if queue_mode:
            results_dir = os.path.join(self.results_dir, 'network_images/')
            if not os.path.isdir(results_dir):
                os.makedirs(results_dir)
            plt.figtext(0.1, 0.05, "total injections: " + str(self.dynetwork._initializations + self.dynetwork._max_initializations))
            plt.figtext(0.1, 0.01, "total drops: " + str(sum(self.dropped_packet_histogram)))
            plt.figtext(0.1, 0.1, f"Episode: {episode} Timeslot: {timestep}")
            plt.savefig(f"{results_dir}dynet_timestep_{timestep}_episode_{episode}.png")
        else:
            plt.savefig(f"{self.results_dir}/network_topology_eps_{episode}_timestep_{timestep}.png")
        plt.clf()

    ''' Handle Packet Delivery Event '''

    def handle_packet_delivery_event(self, dest, packetIdx):
        packet = self.dynetwork._packets.packetList[packetIdx]
        Received_time = packet.get_time()
        # Calculate the true delivery time of this packet
        self.dynetwork._delivery_times.append(Received_time + packet.get_drops()*self.packet_maximal_latency_threshold)
        self.dynetwork._deliveries += 1
        self.arrived_packet_histogram[dest-self.dest_offset] += 1
        # Increase the arrival counter by 1
        self.arrived_packet_counter += 1
        steps_set = set()
        hops = 0
        prevStep = None
        for item in packet._steps:
            step, time = item
            hops += 1
            if step in steps_set:
                self.circles_counter += 1
            else:
                steps_set.add(step)
            if step < self.nnodes:
                self.pathMapping[step][dest - self.dest_offset] += 1
            if prevStep is not None:
                self.PolicyMapping[dest - self.dest_offset][prevStep][step] += 1
            prevStep = step
        self.hops_counter.append(hops)
        # Reset the drop counter of this packet
        packet._drops = 0
        # Add this packet to the
        self.dynetwork.add_free_packet_idx(packetIdx)

    def normalize_reward(self, reward):
        # abs_reward = np.abs(reward)
        # if abs_reward > self.max_reward:
        #     self.max_reward = self.max_reward * (1 - self.alpha) + self.alpha * abs_reward
        # self.avg_reward = self.avg_reward * (1 - self.alpha) + self.alpha * reward
        # reward = float((reward - self.avg_reward) / self.max_reward)
        return reward

class dynetworkEnvQlearning(dynetworkEnvBaseline):
    @staticmethod
    def get_state_space_dim(setting):
        return None

    def __init__(self, seed, algorithm, rewardfun, setting):
        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        super(dynetworkEnvQlearning, self).__init__(algorithm, setting)

        '''current packet, i.e. first item in the dynetwork's packet list'''
        self.td_error_mean = []
        self.td_error_max = []
        self.td_error_min = []
        self.update_reward(rewardfun)

    def updateWhole(self, agent):
        self.updateWholeBaseline()
        learning_transition, rewards = self.router(agent)
        return learning_transition, rewards

    ''' -----------------Q-Learning Functions---------------- '''

    ''' return packet's position and destination'''

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        return (pkt.get_curPos(), pkt.get_endPos())

    def router(self, agent):
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        learning_transitions = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, guarunteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            '''provides pointer for queue of current node'''
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            '''Congestion Measure #1: max queue len'''
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            '''Congestion Measure #2: avg queue len pt1'''
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                ''' Congestion Measure #3: avg percent at capacity'''
                if (queue_size > sending_capacity):
                    '''increment number of nodes that are at capacity'''
                    num_nodes_at_capacity += 1

            '''stores packets which currently have no destination path'''
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = list(self.dynetwork._network.neighbors(nodeIdx))
            for i in range(queue_size):
                '''when node cannot send anymore packets break and move to next node'''
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                pkt_state = self.get_state(copy.deepcopy(self.packet))

                action = agent.act(pkt_state, neighbor_list)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)
                if reward != None:
                    sendctr += 1
                    action_set.add(action)
                    ttl_feature = self.packet_maximal_latency_threshold - self.dynetwork._packets.packetList[self.packet].get_time()
                    if not done:
                        next_state_neighbors = list(self.dynetwork._network.neighbors(action))
                    else:
                        next_state_neighbors = []
                    learning_transitions.append((pkt_state, action, reward, done, neighbor_list, next_state_neighbors))

                    accumulated_reward += reward

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']

        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization,
                        network_throughput, rejections, accumulated_reward)

        ''' Learning Measure #1: Average TD error'''
        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions, accumulated_reward

    """ given an neighboring node (action), will check if node has a available space in that queue. if it does not, the packet stays at current queue. else, packet is sent to action node's queue. """

    def step(self, action, curNode=None):
        reward = None
        done = False

        """ checks if action is None, in which case current node has no neighbors and also checks to see if target node has space in queue """
        if (action == None) or (self.is_capacity(action)):
            ''' In case that the next neighbor returned us a NACK '''
            self.remaining.append(self.dynetwork._packets.packetList[self.packet])
            self.dynetwork._rejections += 1
        else:
            reward, done = self.send_packet(action)

        return reward, done, self.remaining, self.curr_queue, action

    ''' 
    Given next_step, send packet to next_step.
    Check if the node is full/other considerations beforehand. 
    '''
    def send_packet(self, next_step):
        # Extract the packet instance
        pkt = self.dynetwork._packets.packetList[self.packet]
        # Retrieve Current Position
        curr_node = pkt.get_curPos()
        # Retrieve Destination Position
        dest_node = pkt.get_endPos()
        # print(f'action-{next_step}, destination-{dest_node}')
        assert next_step == dest_node or next_step <= self.nnodes
        # Get Edge Weights
        weight = self.dynetwork._network[curr_node][next_step]['edge_delay']
        # Extract the time elapsed in the current BS queue
        curr_time = pkt.get_time()
        if curr_time + weight > self.packet_maximal_latency_threshold:
            self.dynetwork.add_free_packet_idx(self.packet)
            steps_set = set()
            dest = self.dynetwork._packets.packetList[self.packet].get_endPos()
            prevStep = None
            for item in self.dynetwork._packets.packetList[self.packet]._steps:
                step, time = item
                if step in steps_set:
                    self.circles_counter += 1
                else:
                    steps_set.add(step)
                self.dropMapping[step][dest - self.dest_offset] += 1
                if prevStep is not None:
                    self.PolicyMapping[dest - self.dest_offset][prevStep][step] += 1
                prevStep = step

            '''
            Increase the drop counter of this packet
            '''
            self.dynetwork._packets.packetList[self.packet].increase_drops()

            ''' Update the drop packet statistics '''
            self.dropped_packet_histogram[curr_node] += 1

            # Increase the drop counter by 1
            self.dropped_packets_counter += 1
            reward = None
            done = None
        else:
            # Update Packet Next Step
            pkt.set_curPos(next_step)
            reward = -(curr_time - pkt._steps[-1][1])
            # save the next step
            pkt.add_step((next_step, curr_time))
            self.dynetwork._packets.packetList[self.packet].set_time(curr_time + weight)
            if next_step == dest_node:
                """ if packet has reached destination, a new packet is created with the same 'ID' (packet index) but a new destination, which is then redistributed to another node """
                self.handle_packet_delivery_event(dest_node, self.packet)
                reward += self.reward(next_step, dest_node, weight)
                done = True
            else:
                reward += self.reward(next_step, dest_node, weight)
                self.dynetwork._network.nodes[next_step]['receiving_queue'].append((self.packet, weight))
                done = False
        return reward, done

    '''-----------------------------Reward Functions----------------------------'''
    def reward(self, next_step, dest_node, weight):
        try:
            reward = self.reward_func(self, next_step, dest_node, weight)
        except nx.NetworkXNoPath:
            """ if the node the packet was just sent to has no available path to dest_node, we assign a reward of -50 """
            reward = -120
        return reward

    def update_reward(self,rewardfun):
        if rewardfun == 'reward1':
            self.reward_func = reward1
        elif rewardfun == 'reward2':
            self.reward_func = reward2
        elif rewardfun == 'reward3':
            self.reward_func = reward3
        elif rewardfun == 'reward4':
            self.reward_func = reward4
        elif rewardfun == 'reward5':
            self.reward_func = reward5
        elif rewardfun == 'reward6':
            self.reward_func = reward6
        elif rewardfun == 'reward7':
            self.reward_func = reward7
        elif rewardfun == 'reward8':
            self.reward_func = reward8
        elif rewardfun == 'reward9':
            # Return Nothing!
            self.reward_func = lambda env, next_step, dest_node, weight: 0
        else:
            raise Exception('Invalid Reward function for Q-Routing')

    def reset_env(self, rewardfun=None, seed=0, reset_stochastic_engine = False, arrival_rate = None, resetTopology=False):
        super(dynetworkEnvQlearning, self).reset_env(rewardfun=rewardfun, reset_stochastic_engine=reset_stochastic_engine,seed=seed, arrival_rate=arrival_rate, resetTopology=resetTopology)
        if rewardfun is not None:
            self.update_reward(rewardfun)

class dynetworkEnvRandom(dynetworkEnvBaseline):
    @staticmethod
    def get_state_space_dim(setting):
        return None

    def __init__(self, seed, algorithm, rewardfun, setting):
        random.seed(seed)
        np.random.seed(seed)
        super(dynetworkEnvRandom, self).__init__(algorithm, setting)

        '''current packet, i.e. first item in the dynetwork's packet list'''
        self.td_error_mean = []
        self.td_error_max = []
        self.td_error_min = []
        self.update_reward(rewardfun)

    def updateWhole(self, agent):
        self.updateWholeBaseline()
        learning_transition, rewards = self.router(agent)
        return learning_transition, rewards

    ''' -----------------Q-Learning Functions---------------- '''

    ''' return packet's position and destination'''

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        return (pkt.get_curPos(), pkt.get_endPos())

    def router(self, agent):
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        learning_transitions = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, guarunteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            '''provides pointer for queue of current node'''
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            '''Congestion Measure #1: max queue len'''
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            '''Congestion Measure #2: avg queue len pt1'''
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                ''' Congestion Measure #3: avg percent at capacity'''
                if (queue_size > sending_capacity):
                    '''increment number of nodes that are at capacity'''
                    num_nodes_at_capacity += 1

            '''stores packets which currently have no destination path'''
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = list(self.dynetwork._network.neighbors(nodeIdx))
            for i in range(queue_size):
                '''when node cannot send anymore packets break and move to next node'''
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                pkt_state = self.get_state(copy.deepcopy(self.packet))

                action = agent.act(pkt_state, neighbor_list)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)
                if reward != None:
                    sendctr += 1
                    action_set.add(action)
                    learning_transitions.append((pkt_state, action, reward, done))

                    accumulated_reward += reward

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']

        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization,
                        network_throughput, rejections, accumulated_reward)

        ''' Learning Measure #1: Average TD error'''
        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions, accumulated_reward

    """ given an neighboring node (action), will check if node has a available space in that queue. if it does not, the packet stays at current queue. else, packet is sent to action node's queue. """

    def step(self, action, curNode=None):
        reward = None
        done = False

        """ checks if action is None, in which case current node has no neighbors and also checks to see if target node has space in queue """
        if (action == None) or (self.is_capacity(action)):
            ''' In case that the next neighbor returned us a NACK '''
            self.remaining.append(self.dynetwork._packets[self.packet])
            self.dynetwork._rejections += 1
        else:
            reward, done = self.send_packet(action)

        return reward, done, self.remaining, self.curr_queue, action

    ''' 
    Given next_step, send packet to next_step.
    Check if the node is full/other considerations beforehand. 
    '''
    def send_packet(self, next_step):
        # Extract the packet instance
        pkt = self.dynetwork._packets.packetList[self.packet]
        # Retrieve Current Position
        curr_node = pkt.get_curPos()
        # Retrieve Destination Position
        dest_node = pkt.get_endPos()
        # print(f'action-{next_step}, destination-{dest_node}')
        if next_step >= self.nnodes:
            assert next_step == dest_node
        # Get Edge Weights
        weight = self.dynetwork._network[curr_node][next_step]['edge_delay']
        error_prob = self.dynetwork._network[curr_node][next_step]['edge_error_probability']
        # Update Packet Next Step
        pkt.set_curPos(next_step)
        # Extract the time elapsed in the current BS queue
        curr_time = pkt.get_time()
        reward = -(curr_time - pkt._steps[-1][1])
        # save the next step
        pkt.add_step((next_step, curr_time))
        self.dynetwork._packets.packetList[self.packet].set_time(curr_time + weight)
        if next_step == dest_node:
            """ if packet has reached destination, a new packet is created with the same 'ID' (packet index) but a new destination, which is then redistributed to another node """
            self.handle_packet_delivery_event(dest_node, self.packet)
            reward += self.reward(next_step, dest_node, weight)
            done = True
        else:
            reward += self.reward(next_step, dest_node, weight)
            self.dynetwork._network.nodes[next_step]['receiving_queue'].append((self.packet, weight))
            done = False
        return reward, done

    '''-----------------------------Reward Functions----------------------------'''
    def reward(self, next_step, dest_node, weight):
        try:
            reward = self.reward_func(self, next_step, dest_node, weight)
        except nx.NetworkXNoPath:
            """ if the node the packet was just sent to has no available path to dest_node, we assign a reward of -50 """
            reward = -120
        return reward

    def update_reward(self,rewardfun):
        if rewardfun == 'reward1':
            self.reward_func = reward1
        elif rewardfun == 'reward2':
            self.reward_func = reward2
        elif rewardfun == 'reward3':
            self.reward_func = reward3
        elif rewardfun == 'reward4':
            self.reward_func = reward4
        elif rewardfun == 'reward5':
            self.reward_func = reward5
        elif rewardfun == 'reward6':
            self.reward_func = reward6
        elif rewardfun == 'reward7':
            self.reward_func = reward7
        elif rewardfun == 'reward8':
            self.reward_func = reward8
        elif rewardfun == 'reward9':
            # Return Nothing!
            self.reward_func = lambda env, next_step, dest_node, weight: 0
        else:
            raise Exception('Invalid Reward function for Q-Routing')

    def reset_env(self, rewardfun=None, seed=0, reset_stochastic_engine = False, arrival_rate = None, resetTopology=False):
        super(dynetworkEnvRandom, self).reset_env(rewardfun=rewardfun, reset_stochastic_engine=reset_stochastic_engine,seed=seed, arrival_rate=arrival_rate, resetTopology=resetTopology)
        if rewardfun is not None:
            self.update_reward(rewardfun)

class dynetworkEnvQlearningWithTimeStep(dynetworkEnvQlearning):
    @staticmethod
    def get_state_space_dim(setting):
        return None

    def __init__(self, seed, algorithm, rewardfun, setting):
        super(dynetworkEnvQlearningWithTimeStep, self).__init__(seed,algorithm,rewardfun,setting)
    ''' return packet's position and destination'''

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        return (pkt.get_curPos(), pkt.get_endPos(), pkt._steps[-1][1])

class dynetworkEnvFullEchoQLearning(dynetworkEnvQlearning):
    def __init__(self, seed, algorithm, rewardfun, setting):
        super().__init__(seed, algorithm, rewardfun, setting)
        self.availableBs = np.arange(start=0, stop=self.nnodes)

    def calculate_neighbors_rewards(self, neighbors):
        # Calculate information regarding the next nodes rewards since we are using full echo!
        pkt = self.dynetwork._packets.packetList[self.packet]
        # Retrieve Current Position
        src = pkt.get_curPos()
        # Retrieve Destination Position
        dst = pkt.get_endPos()
        # Extract the time elapsed in the current BS queue
        curr_time = pkt.get_time()
        base_reward = -(curr_time - pkt._steps[-1][1])
        rewards = []
        dones = []
        availableNeighbors = []
        nextStateNeighbors = []
        for next_step in neighbors:
            if next_step in self.availableBs or next_step == dst:
                weight = self.dynetwork._network[src][next_step]['edge_delay']
                temp_reward = base_reward + self.reward(next_step, dst, weight)
                # Update the current neighbor reward function
                rewards.append(temp_reward)
                # Check if we arrived to the destination
                dones.append(dst == next_step)
                availableNeighbors.append(next_step)
                nextStateNeighbors.append(list(self.dynetwork._network.neighbors(next_step)))

        if len(rewards) == 0:
            rewards = None
        return rewards, dones, availableNeighbors, nextStateNeighbors

    def router(self, agent):
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        learning_transitions = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, guarunteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            '''provides pointer for queue of current node'''
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            '''Congestion Measure #1: max queue len'''
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            '''Congestion Measure #2: avg queue len pt1'''
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                ''' Congestion Measure #3: avg percent at capacity'''
                if (queue_size > sending_capacity):
                    '''increment number of nodes that are at capacity'''
                    num_nodes_at_capacity += 1

            '''stores packets which currently have no destination path'''
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = list(self.dynetwork._network.neighbors(nodeIdx))
            for i in range(queue_size):
                '''when node cannot send anymore packets break and move to next node'''
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                pkt_state = self.get_state(copy.deepcopy(self.packet))

                action = agent.act(pkt_state, neighbor_list)
                neighbors_rewards, dones, neighbor_list, next_state_neighbors = self.calculate_neighbors_rewards(neighbor_list)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)
                if reward != None:
                    sendctr += 1
                    action_set.add(action)
                    ttl_feature = self.packet_maximal_latency_threshold - self.dynetwork._packets.packetList[self.packet].get_time()
                    learning_transitions.append(((pkt_state, neighbor_list), action, neighbors_rewards, dones, neighbor_list, next_state_neighbors))

                    accumulated_reward += reward

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']

        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization,
                        network_throughput, rejections, accumulated_reward)

        ''' Learning Measure #1: Average TD error'''
        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions, accumulated_reward

class dynetworkEnvDeterminstic(dynetworkEnvBaseline):
    def __init__(self, seed, algorithm, setting):
        random.seed(seed)
        np.random.seed(seed)
        super(dynetworkEnvDeterminstic, self).__init__(algorithm=algorithm, setting=setting)
        self.delay = 'global_delay' if algorithm == 'Shortest-Path' else 'delay'

    def updateWhole(self, agent):
        self.agent = agent
        self.updateWholeBaseline()
        self.router(self.router_type, self.delay)

    def router(self, router_type='dijkstra', weight='delay'):
        raise Exception('Invalid method for this dynetworkEnvDeterminstic class!')

    def handle_node_packet(self, curr_queue, remaining, router_type, weight):
        packetIdx = curr_queue.pop()
        packetInstance = self.dynetwork._packets.packetList[packetIdx]
        currPos = packetInstance.get_curPos()
        destPos = packetInstance.get_endPos()
        sent = False
        next_step = None
        # Raise Exception in case a packet arrived to her destination with out our notice
        assert currPos != destPos
        try:
            next_step = self.get_next_step(currPos, destPos, router_type, weight)
            if self.is_capacity(next_step):
                remaining.append(packetInstance)
                self.dynetwork._rejections += 1
            else:
                self.send_packet(packetInstance.get_index(), currPos, next_step)
                sent = True
        except (nx.NetworkXNoPath, KeyError):
            remaining.append(packetInstance)
        return remaining, curr_queue, sent, next_step

    '''return the node for packet to route to in the next step using shortest path algorithm'''

    def get_next_step(self, currPos, destPos, router_type, weight):
        raise Exception('This is not an invalid method for this class')

    '''helper function to route one packet'''

    def send_packet(self, pkt, curr, next_step):
        # Get current packet time
        curr_time = self.dynetwork._packets.packetList[pkt].get_time()

        # add the step to the packet path
        self.dynetwork._packets.packetList[pkt].add_step((next_step, curr_time))
        # Update packet next position according to next step
        self.dynetwork._packets.packetList[pkt].set_curPos(next_step)
        # Extract edge weights
        weight = self.dynetwork._network[curr][next_step]['edge_delay']
        # Update time of the packet
        self.dynetwork._packets.packetList[pkt].set_time(curr_time + weight)
        # Extract packet destination
        dest = self.dynetwork._packets.packetList[pkt].get_endPos()
        # Verify that we are not using the users as relays
        if next_step >= self.nnodes:
            assert next_step == dest
        # Check if the packet arrived to her destination
        if self.dynetwork._packets.packetList[pkt].get_curPos() == dest:
            self.handle_packet_delivery_event(dest,pkt)
        else:
            self.dynetwork._network.nodes[next_step]['receiving_queue'].append((pkt, weight))

class dynetworkEnvShortestPath(dynetworkEnvDeterminstic):
    def __init__(self, seed, algorithm, setting, rewardfun=None):
        assert algorithm == 'Shortest-Path'
        super(dynetworkEnvShortestPath, self).__init__(seed=seed,algorithm='Shortest-Path', setting=setting)
        self.preds = None

    def router(self, router_type='dijkstra', weight='delay'):
        if str.lower(router_type) != 'dijkstra':
            if weight == 'delay':
                self.preds, _ = nx.floyd_warshall_predecessor_and_distance(self.dynetwork._network, weight='edge_delay')
            elif weight == 'global_delay':
                self.preds, _ = nx.floyd_warshall_predecessor_and_distance(self.dynetwork._network, weight='edge_global_delay')
            else:
                self.preds, _ = nx.floyd_warshall_predecessor_and_distance(self.dynetwork._network)
        node_queue_lengths = np.zeros((self.nnodes))
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        max_queue_length = 0
        rejections = 0

        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            '''provides pointer for queue of current node'''
            curr_queue = self.dynetwork._network.nodes[nodeIdx]['sending_queue']
            sending_capacity = self.dynetwork._network.nodes[nodeIdx]['max_send_capacity']
            queue_size = len(curr_queue)



            '''Congestion Measure #1: max queue length'''
            if (queue_size > max_queue_length):
                max_queue_length = queue_size
            '''Congestion Measure #2: average queue length'''
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1

                '''Congestion Measure #3: average percentage of active nodes at capacity'''
                if (queue_size > sending_capacity):
                    num_nodes_at_capacity += 1

            '''stores packets which currently have no path to destination'''
            remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            for i in range(queue_size):
                '''when node cannot send anymore packets, break and move on to next node'''
                if sendctr == sending_capacity:
                    rejections += (len(self.dynetwork._network.nodes[nodeIdx]['sending_queue']))
                    break
                remaining, curr_queue, sent, next_step = self.handle_node_packet(curr_queue, remaining, router_type, weight)
                if sent:
                    sendctr += 1
                    action_set.add(next_step)

            neighbors_number = len(list(self.dynetwork._network.neighbors(nodeIdx)))
            self.dynetwork._network.nodes[nodeIdx]['sending_queue'] = remaining + self.dynetwork._network.nodes[nodeIdx]['sending_queue']

            """ Congestion Measure #5: Link Utilization Percentile """
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)
        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput,
                        rejections, 0)

    '''return the node for packet to route to in the next step using shortest path algorithm'''
    def get_next_step(self, currPos, destPos, router_type, weight):
        if str.lower(router_type) == 'dijkstra' and weight == 'delay':
            return nx.dijkstra_path(self.dynetwork._network, currPos, destPos, weight='edge_delay')[1]
        elif str.lower(router_type) == 'dijkstra' and weight == 'global_delay':
            return nx.dijkstra_path(self.dynetwork._network, currPos, destPos, weight='edge_global_delay')[1]
        elif str.lower(router_type) == 'dijkstra':
            return nx.dijkstra_path(self.dynetwork._network, currPos, destPos)[1]
        else:
            return nx.reconstruct_path(currPos, destPos, self.preds)[1]

class dynetworkEnvBackPressure(dynetworkEnvDeterminstic):
    def __init__(self, seed, algorithm, setting, rewardfun=None):
        assert algorithm == 'Back-Pressure'
        super(dynetworkEnvBackPressure, self).__init__(seed=seed,algorithm='Back-Pressure', setting=setting)

    def router(self, router_type='BackPressure', weight='delay'):
        node_queue_lengths = np.zeros((self.nnodes))
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0
        max_queue_length = 0
        rejections = 0

        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []

        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            '''provides pointer for queue of current node'''
            curr_queue = self.dynetwork._network.nodes[nodeIdx]['sending_queue']
            sending_capacity = self.dynetwork._network.nodes[nodeIdx]['max_send_capacity']
            queue_size = len(curr_queue)

            '''Congestion Measure #1: max queue length'''
            if (queue_size > max_queue_length):
                max_queue_length = queue_size
            '''Congestion Measure #2: average queue length'''
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1

                '''Congestion Measure #3: average percentage of active nodes at capacity'''
                if (queue_size > sending_capacity):
                    num_nodes_at_capacity += 1

            ''' stores packets which currently have no path to destination '''
            remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbors = list(self.dynetwork._network.neighbors(nodeIdx))
            neighbors_number = len(neighbors)
            for i in range(queue_size):
                '''when node cannot send anymore packets, break and move on to next node'''
                if sendctr == sending_capacity:
                    rejections += (len(self.dynetwork._network.nodes[nodeIdx]['sending_queue']))
                    break
                remaining, curr_queue, sent, next_step = self._handle_node_packet(nodeIdx, curr_queue, remaining, neighbors)
                if sent:
                    sendctr += 1
                    action_set.add(next_step)

            self.dynetwork._network.nodes[nodeIdx]['sending_queue'] = remaining + self.dynetwork._network.nodes[nodeIdx]['sending_queue']

            """ Congestion Measure #5: Link Utilization Percentile """
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)
        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput,
                        rejections, 0)

    '''helper function to move packets to their corresponding queues'''

    def _handle_node_packet(self, currPos, curr_queue, remaining, neighbors):
        next_step, packet = self.agent.act(self.dynetwork, currPos, neighbors)

        if next_step is None or packet is None:
            return remaining,curr_queue,False,None

        currPos = self.dynetwork._packets.packetList[packet].get_curPos()
        destPos = self.dynetwork._packets.packetList[packet].get_endPos()
        sent = False
        if currPos == destPos:
            curr_queue.remove(packet)
        else:
            if next_step is None or self.is_capacity(next_step):
                curr_queue.remove(packet)
                remaining.append(packet)
                self.dynetwork._rejections += 1
            else:
                self.send_packet(packet, currPos, next_step)
                curr_queue.remove(packet)
                sent = True
        return remaining, curr_queue, sent, next_step

    '''helper function to route one packet '''

    def send_packet(self, pkt, curr, next_step):
        # Get packet's current time
        curr_time = self.dynetwork._packets.packetList[pkt].get_time()
        # add the step to the packet path
        self.dynetwork._packets.packetList[pkt].add_step((next_step, curr_time))
        # Update packet next current position
        self.dynetwork._packets.packetList[pkt].set_curPos(next_step)
        # Extract from the graph the corrsponding edge delay
        weight = self.dynetwork._network[curr][next_step]['edge_delay']
        # Update packet's time
        self.dynetwork._packets.packetList[pkt].set_time(curr_time + weight)
        # Extract packet destination
        dest = self.dynetwork._packets.packetList[pkt].get_endPos()

        if self.dynetwork._packets.packetList[pkt].get_curPos() == dest:
            self.handle_packet_delivery_event(dest, pkt)
        else:
            self.dynetwork._network.nodes[next_step]['receiving_queue'].append((pkt, weight))

class dynetworkEnvCentralizedFuncApproximation(dynetworkEnvQlearning):
    '''Initialization of the network'''
    @staticmethod
    def get_state_space_dim(setting):
        numBs = setting["NETWORK"]["number Basestation"]
        numUsers = setting["NETWORK"]["number user"]

        if numUsers == 0:
            destSize = numBs
        else:
            destSize = numUsers

        return numBs + destSize + (numBs + numUsers)

    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvCentralizedFuncApproximation, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        # Calculate the current destination size
        self.destSize = self.nnodes if self.num_user == 0 else self.num_user
        if self.num_user != 0:
            self.destOneHot = torch.eye((self.num_user))
        else:
            self.destOneHot = torch.eye((self.nnodes))
        self.srcOneHot = torch.eye((self.nnodes))
        self.neighborhood_vec = torch.zeros((self.nnodes, self.nnodes + self.num_user))
        for nodeIdx in range(self.nnodes):
            self.neighborhood_vec[nodeIdx][list(self.dynetwork._network.neighbors(nodeIdx))] = 1
        self.state_dim = dynetworkEnvCentralizedFuncApproximation.get_state_space_dim(setting)

    ''' 
    Function to handle routing all the packets in one time step. 
    Set will_learn to True if we are training and wish to update the Q-table; 
    else if we are testing set will_learn = False. 
    '''

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        src, dest = pkt.get_curPos(), pkt.get_endPos()
        if src == dest:
            return torch.zeros((1, self.state_dim)), src, dest
        state = torch.unsqueeze(torch.concat((self.neighborhood_vec[src], self.srcOneHot[src], self.destOneHot[dest-self.dest_offset])), dim=0)
        return state, src, dest

    def router(self, agent):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0

        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1

            """ stores packets which currently have no destination path """
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = sorted(list(self.dynetwork._network.neighbors(nodeIdx)))
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                cur_state, src, destination = self.get_state(self.packet)

                """ whether or not we input nodes' queue_size to the network """
                action = agent.act(cur_state, neighbor_list, destination)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)

                if action != None:
                    next_state, src, destination = self.get_state(self.packet)
                    ''' Check if this was a successful action and we indeed received a reward '''
                    if reward != None:
                        sendctr += 1
                        action_set.add(action)
                        done = action == destination
                        accumulated_reward += reward
                        agent.dqn.replay_memory.push(cur_state, action, next_state, reward, done, destination, nodeIdx)

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']


        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, accumulated_reward)

        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return accumulated_reward

    def updateWhole(self, agent):
        self.updateWholeBaseline()
        rewards = self.router(agent)
        return rewards

    '''Initialize all neural networks with one neural network initialized for each node in the network.'''

class dynetworkEnvA2cCentralized(dynetworkEnvCentralizedFuncApproximation):
    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvA2cCentralized, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        self.state_dim = dynetworkEnvCentralizedFuncApproximation.get_state_space_dim(setting)

    def router(self, agent):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0

        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        learning_transitions = []
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1

            """ stores packets which currently have no destination path """
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = sorted(list(self.dynetwork._network.neighbors(nodeIdx)))
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                cur_state, src, destination = self.get_state(self.packet)

                """ whether or not we input nodes' queue_size to the network """
                action = agent.act(cur_state, neighbor_list, destination)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)

                if action != None and not self.is_capacity(action):
                    next_state, src, destination = self.get_state(self.packet)
                    ''' Check if this was a successful action and we indeed received a reward '''
                    if reward != None:
                        sendctr += 1
                        action_set.add(action)
                        done = action == destination
                        accumulated_reward += reward
                        learning_transitions.append((cur_state, action, reward, next_state, done, neighbor_list, destination))

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']


        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, accumulated_reward)

        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions

class dynetworkEnvA2cDeCentralized(dynetworkEnvCentralizedFuncApproximation):
    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvA2cDeCentralized, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        self.state_dim = dynetworkEnvA2cDeCentralized.get_state_space_dim(setting)

    @staticmethod
    def get_state_space_dim(setting):
        numBs = setting["NETWORK"]["number Basestation"]
        numUsers = setting["NETWORK"]["number user"]

        if numUsers == 0:
            destSize = numBs
        else:
            destSize = numUsers

        return destSize + (numBs + numUsers)

    def router(self, agent):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0

        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        learning_transitions = []
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1

            """ stores packets which currently have no destination path """
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = sorted(list(self.dynetwork._network.neighbors(nodeIdx)))
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                cur_state, src, destination = self.get_state(self.packet)

                """ whether or not we input nodes' queue_size to the network """
                action = agent.act(nodeIdx, cur_state, neighbor_list, destination)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)

                if action != None and not self.is_capacity(action):
                    next_state, src, destination = self.get_state(self.packet)
                    ''' Check if this was a successful action and we indeed received a reward '''
                    if reward != None:
                        sendctr += 1
                        action_set.add(action)
                        done = action == destination
                        accumulated_reward += reward
                        learning_transitions.append((cur_state, action, reward, next_state, done, neighbor_list, destination, nodeIdx))

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']


        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, accumulated_reward)

        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions

    def get_state(self, pktIdx):
        pkt = self.dynetwork._packets.packetList[self.packet]
        src, dest = pkt.get_curPos(), pkt.get_endPos()
        if src == dest:
            return torch.zeros((1, self.state_dim)), src, dest
        state = torch.unsqueeze(torch.concat((self.neighborhood_vec[src], self.destOneHot[dest-self.dest_offset])), dim=0)
        return state, src, dest

class dynetworkEnvCentralizedFuncApproximationRelationalState(dynetworkEnvCentralizedFuncApproximation):
    '''Initialization of the network'''
    @staticmethod
    def get_state_space_dim(setting):
        numBs = setting["NETWORK"]["number Basestation"]
        numUsers = setting["NETWORK"]["number user"]

        return numBs + numBs + numBs + 2

    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvCentralizedFuncApproximationRelationalState, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        self.state_dim = dynetworkEnvCentralizedFuncApproximationRelationalState.get_state_space_dim(setting)
        self.user_mapping_one_hot = torch.zeros((self.num_user, self.nnodes))
        self.neighborhood_vec = torch.zeros((self.nnodes, self.nnodes))
        for nodeIdx in range(self.nnodes):
            neighbors_list = list(self.dynetwork._network.neighbors(nodeIdx))
            for dest in range(self.num_user):
                offseted_dest = dest + self.dest_offset
                if offseted_dest in neighbors_list:
                    self.user_mapping_one_hot[dest][nodeIdx] = 1

            temp_bs_neighbors_list = [neighbor for neighbor in neighbors_list if (neighbor < self.nnodes)]
            self.neighborhood_vec[nodeIdx][temp_bs_neighbors_list] = 1
    ''' 
    Function to handle routing all the packets in one time step. 
    Set will_learn to True if we are training and wish to update the Q-table; 
    else if we are testing set will_learn = False. 
    '''
    def get_state(self, pktIdx):
        def get_packet_features():
            ttl_feature = self.packet_maximal_latency_threshold - pkt.get_time()
            return [ttl_feature]

        def get_local_device_features(idx):
            queue_length = len(self.dynetwork._network.nodes[idx]["sending_queue"])
            shortest_distance_to_dest = self.optimalNumHops[idx][dest]
            return np.array([queue_length, shortest_distance_to_dest])

        def get_neighborhood_device_features():
            avialable_neighbors = list(np.squeeze(np.argwhere(self.neighborhood_vec[src] != 0)))
            neighbors_features = []
            for neighbor in avialable_neighbors:
                neighbor_features = np.array(get_local_device_features(neighbor.item()))
                neighbors_features.append(neighbor_features)

            neighbors_features = np.array(neighbors_features)
            aggregated_features = np.concatenate((neighbors_features.max(axis=0), neighbors_features.min(axis=0), neighbors_features.mean(axis=0)))
            return aggregated_features

        pkt = self.dynetwork._packets.packetList[self.packet]
        src, dest = pkt.get_curPos(), pkt.get_endPos()
        if src == dest:
            return torch.zeros((1, self.state_dim)), src, dest
        state = torch.unsqueeze(torch.concat((self.neighborhood_vec[src], self.srcOneHot[src], self.user_mapping_one_hot[dest-self.dest_offset])), dim=0)
        pkt_feat = get_packet_features()
        local_feat = get_local_device_features(src)
        neighborhood_feat = get_neighborhood_device_features()

        local_features = np.concatenate((pkt_feat, neighborhood_feat))
        state = torch.unsqueeze(torch.concat((torch.squeeze(state), torch.Tensor(local_features))), dim=0)
        return state, src, dest

    def router(self, agent):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        td_error = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0

        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1

            """ stores packets which currently have no destination path """
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = sorted(list(self.dynetwork._network.neighbors(nodeIdx)))
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                cur_state, src, destination = self.get_state(self.packet)

                """ whether or not we input nodes' queue_size to the network """
                action = agent.act(cur_state, neighbor_list, destination)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)

                if action != None:
                    next_state, src, destination = self.get_state(self.packet)
                    ''' Check if this was a successful action and we indeed received a reward '''
                    if reward != None:
                        sendctr += 1
                        action_set.add(action)
                        ''' Check if this was a successful action and we indeed received a reward '''
                        if action < self.nnodes:
                            done = self.user_mapping_one_hot[destination-self.dest_offset][action] == 1
                            if done:
                                # Get Edge Weights
                                weight = self.dynetwork._network[action][destination]['edge_delay']
                                # Calculate the reward till the end minus the 1 timeslot that we transmit the packet
                                reward += reward7(self, action, destination, weight) - 1
                            agent.dqn.replay_memory.push(cur_state, action, next_state, reward, done, destination, nodeIdx)
                        accumulated_reward += reward

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']


        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, accumulated_reward)

        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return accumulated_reward

class dynetworkEnvCentralizedA2cRelationalState(dynetworkEnvCentralizedFuncApproximation):
    '''Initialization of the network'''
    @staticmethod
    def get_state_space_dim(setting):
        numBs = setting["NETWORK"]["number Basestation"]

        return numBs + numBs + numBs + 2

    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvCentralizedA2cRelationalState, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        self.state_dim = dynetworkEnvCentralizedFuncApproximationRelationalState.get_state_space_dim(setting)
        self.neighborhood_vec = torch.zeros((self.nnodes, self.nnodes))
        for nodeIdx in range(self.nnodes):
            self.neighborhood_vec[nodeIdx][[node for node in list(self.dynetwork._network.neighbors(nodeIdx)) if node < self.nnodes]] = 1

    ''' 
    Function to handle routing all the packets in one time step. 
    Set will_learn to True if we are training and wish to update the Q-table; 
    else if we are testing set will_learn = False. 
    '''
    def get_state(self, pktIdx):
        def get_packet_features():
            ttl_feature = self.packet_maximal_latency_threshold - pkt.get_time()
            return [ttl_feature]

        def get_local_device_features(idx):
            queue_length = len(self.dynetwork._network.nodes[idx]["sending_queue"])
            shortest_distance_to_dest = self.optimalNumHops[idx][dest]
            return np.array([queue_length, shortest_distance_to_dest])

        def get_neighborhood_device_features():
            avialable_neighbors = list(np.squeeze(np.argwhere(self.neighborhood_vec[src] != 0)))
            neighbors_features = []
            for neighbor in avialable_neighbors:
                neighbor_features = np.array(get_local_device_features(neighbor.item()))
                neighbors_features.append(neighbor_features)

            neighbors_features = np.array(neighbors_features)
            aggregated_features = np.concatenate((neighbors_features.max(axis=0), neighbors_features.min(axis=0), neighbors_features.mean(axis=0)))
            return aggregated_features

        pkt = self.dynetwork._packets.packetList[self.packet]
        src, dest = pkt.get_curPos(), pkt.get_endPos()
        if src == dest:
            return torch.zeros((1, self.state_dim)), src, dest
        state = torch.unsqueeze(torch.concat((self.neighborhood_vec[src], self.srcOneHot[src], self.user_association_rule_one_hot[dest-self.dest_offset])), dim=0)


        pkt_feat = get_packet_features()
        local_feat = get_local_device_features(src)
        neighborhood_feat = get_neighborhood_device_features()

        # local_features = np.concatenate((pkt_feat, local_feat, neighborhood_feat))
        # local_features = np.concatenate((pkt_feat, local_feat))
        # local_features = np.concatenate((pkt_feat, neighborhood_feat))
        # local_features = np.concatenate((local_feat, neighborhood_feat))
        # local_features = neighborhood_feat
        # local_features = pkt_feat
        local_features = local_feat
        # local_features = np.concatenate((pkt_feat, local_feat, neighborhood_feat))
        state = torch.unsqueeze(torch.concat((torch.squeeze(state), torch.Tensor(local_features))), dim=0)
        return state, src, dest

    def router(self, agent):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        td_error = []
        learning_transitions = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0

        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1

            """ stores packets which currently have no destination path """
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = sorted(list(self.dynetwork._network.neighbors(nodeIdx)))
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                cur_state, src, destination = self.get_state(self.packet)

                """ whether or not we input nodes' queue_size to the network """
                action = agent.act(cur_state, neighbor_list, destination)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)

                if action != None:
                    next_state, src, destination = self.get_state(self.packet)
                    ''' Check if this was a successful action and we indeed received a reward '''
                    if reward != None:
                        sendctr += 1
                        action_set.add(action)
                        ''' Check if this was a successful action and we indeed received a reward '''
                        try:
                            if action < self.nnodes:
                                done = self.user_association_rule_one_hot[destination-self.dest_offset][action] == 1
                                if done:
                                    # Get Edge Weights
                                    weight = self.dynetwork._network[action][destination]['edge_delay']
                                    # Calculate the reward till the end minus the 1 timeslot that we transmit the packet
                                    reward += reward7(self, action, destination, weight) - 1

                                learning_transitions.append((cur_state, action, reward, next_state, done, neighbor_list, destination))
                        except TypeError:
                            partial_action = action[0]
                            if partial_action < self.nnodes:
                                done = self.user_association_rule_one_hot[destination - self.dest_offset][partial_action] == 1
                                if done:
                                    # Get Edge Weights
                                    weight = self.dynetwork._network[partial_action][destination]['edge_delay']
                                    # Calculate the reward till the end minus the 1 timeslot that we transmit the packet
                                    reward += reward7(self, partial_action, destination, weight) - 1

                                learning_transitions.append((cur_state, action, reward, done, neighbor_list, sorted(list(self.dynetwork._network.neighbors(action[0])))))
                        accumulated_reward += reward

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']


        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, accumulated_reward)

        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions

class dynetworkEnvDeCentralizedA2cRelationalState(dynetworkEnvA2cDeCentralized):
    '''Initialization of the network'''
    @staticmethod
    def get_state_space_dim(setting):
        numBs = setting["NETWORK"]["number Basestation"]
        return numBs + numBs + 5

    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvDeCentralizedA2cRelationalState, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        self.state_dim = dynetworkEnvCentralizedFuncApproximationRelationalState.get_state_space_dim(setting)
        self.neighborhood_vec = torch.zeros((self.nnodes, self.nnodes))
        for nodeIdx in range(self.nnodes):
            self.neighborhood_vec[nodeIdx][[node for node in list(self.dynetwork._network.neighbors(nodeIdx)) if node < self.nnodes]] = 1


    ''' 
    Function to handle routing all the packets in one time step. 
    Set will_learn to True if we are training and wish to update the Q-table; 
    else if we are testing set will_learn = False. 
    '''
    def get_state(self, pktIdx):
        def get_packet_features():
            ttl_feature = self.packet_maximal_latency_threshold - pkt.get_time()
            return [ttl_feature]

        def get_local_device_features(idx):
            queue_length = len(self.dynetwork._network.nodes[idx]["sending_queue"])
            return np.array([queue_length])

        def get_neighborhood_device_features():
            avialable_neighbors = list(np.squeeze(np.argwhere(self.neighborhood_vec[src] != 0)))
            neighbors_features = []
            for neighbor in avialable_neighbors:
                neighbor_features = np.array(get_local_device_features(neighbor.item()))
                neighbors_features.append(neighbor_features)

            neighbors_features = np.array(neighbors_features)
            aggregated_features = np.concatenate((neighbors_features.max(axis=0), neighbors_features.min(axis=0), neighbors_features.mean(axis=0)))
            return aggregated_features

        pkt = self.dynetwork._packets.packetList[self.packet]
        src, dest = pkt.get_curPos(), pkt.get_endPos()
        if src == dest:
            return torch.zeros((1, self.state_dim)), src, dest
        state = torch.unsqueeze(torch.concat((self.neighborhood_vec[src], self.user_association_rule_one_hot[dest-self.dest_offset])), dim=0)
        pkt_feat = get_packet_features()
        local_feat = get_local_device_features(src)
        neighborhood_feat = get_neighborhood_device_features()
        local_features = np.concatenate((pkt_feat, local_feat, neighborhood_feat))
        state = torch.unsqueeze(torch.concat((torch.squeeze(state), torch.Tensor(local_features))), dim=0)
        return state, src, dest

    def router(self, agent):
        """ router attempts to route as many packets as the network will allow """
        node_queue_lengths = np.zeros((self.nnodes))
        link_utilization = {'Links': [], 'availableLinks': []}
        network_throughput = []
        td_error = []
        learning_transitions = []
        num_nodes_at_capacity = 0
        num_nonEmpty_nodes = 0

        max_queue_length = 0
        rejections = 0
        accumulated_reward = 0
        '''iterate all nodes'''
        for nodeIdx in range(self.nnodes):
            """ the self.nodes_traversed tracks the number of nodes we have looped over, 
            guaranteeing that each packet will have the same epsilon at each time step"""
            self.nodes_traversed += 1
            if self.nodes_traversed == self.nnodes:
                agent.config['update_epsilon'] = True
                self.nodes_traversed = 0
            node = self.dynetwork._network.nodes[nodeIdx]
            """ provides pointer for queue of current node """
            self.curr_queue = node['sending_queue']
            sending_capacity = node['max_send_capacity']
            queue_size = len(self.curr_queue)

            """ Congestion Measure #1: maximum queue lengths """
            if (queue_size > max_queue_length):
                max_queue_length = queue_size

            """ Congestion Measure #2: avg queue len pt1 """
            if (queue_size > 0):
                node_queue_lengths[nodeIdx] = queue_size
                num_nonEmpty_nodes += 1
                """ Congestion Measure #3: avg percent at capacity """
                if (queue_size > sending_capacity):
                    """ increment number of nodes that are at capacity """
                    num_nodes_at_capacity += 1

            """ stores packets which currently have no destination path """
            self.remaining = PriorityQueue(self.max_queue, is_fifo=self.setting["NETWORK"]["is_fifo"])
            sendctr = 0
            action_set = set()
            neighbor_list = sorted(list(self.dynetwork._network.neighbors(nodeIdx)))
            for i in range(queue_size):
                """ when node cannot send anymore packets break and move to next node """
                if sendctr == sending_capacity:
                    rejections += (1 * (len(node['sending_queue'])))
                    break
                self.packet = self.curr_queue.pop()
                cur_state, src, destination = self.get_state(self.packet)

                """ whether or not we input nodes' queue_size to the network """
                action = agent.act(nodeIdx, cur_state, neighbor_list, destination)
                reward, done, self.remaining, self.curr_queue, action = self.step(action, nodeIdx)
                if action != None:
                    next_state, src, destination = self.get_state(self.packet)
                    ''' Check if this was a successful action and we indeed received a reward '''
                    if reward != None:
                        sendctr += 1
                        action_set.add(action)
                        ''' Check if this was a successful action and we indeed received a reward '''
                        if action < self.nnodes:
                            done = self.user_association_rule_one_hot[destination-self.dest_offset][action] == 1
                            if done:
                                # Get Edge Weights
                                weight = self.dynetwork._network[action][destination]['edge_delay']
                                # Calculate the reward till the end minus the 1 timeslot that we transmit the packet
                                reward += reward7(self, action, destination, weight) - 1
                            learning_transitions.append((cur_state, action, reward, next_state, done, neighbor_list, destination, nodeIdx))
                        accumulated_reward += reward

            """ Congestion Measure #5: Link Utilization Percentile """
            neighbors_number = len(neighbor_list)
            if neighbors_number != 0:
                link_utilization['Links'].append(len(action_set))
                link_utilization['availableLinks'].append(sending_capacity)

            """ Congestion Measure #6: Network Throughput """
            network_throughput.append(sendctr/sending_capacity)

            node['sending_queue'] = self.remaining + node['sending_queue']


        '''
        Update stats
        '''
        self.save_stats(max_queue_length, node_queue_lengths, num_nodes_at_capacity, num_nonEmpty_nodes, link_utilization, network_throughput, rejections, accumulated_reward)

        if len(td_error) != 0:
            self.td_error_mean.append(np.average(td_error))
            self.td_error_max.append(np.max(td_error))
            self.td_error_min.append(np.min(td_error))
        else:
            self.td_error_mean.append(0)
            self.td_error_max.append(0)
            self.td_error_min.append(0)
        return learning_transitions

class dynetworkEnvFedCentralizedA2cRelationalState(dynetworkEnvDeCentralizedA2cRelationalState):
    '''Initialization of the network'''
    @staticmethod
    def get_state_space_dim(setting):
        numBs = setting["NETWORK"]["number Basestation"]
        return numBs + numBs + numBs + 5

    def __init__(self, algorithm, setting, seed, rewardfun):
        super(dynetworkEnvFedCentralizedA2cRelationalState, self).__init__(algorithm=algorithm, rewardfun=rewardfun, seed=seed, setting=setting)
        self.state_dim = dynetworkEnvFedCentralizedA2cRelationalState.get_state_space_dim(setting)

    ''' 
    Function to handle routing all the packets in one time step. 
    Set will_learn to True if we are training and wish to update the Q-table; 
    else if we are testing set will_learn = False. 
    '''
    def get_state(self, pktIdx):
        def get_packet_features():
            ttl_feature = self.packet_maximal_latency_threshold - pkt.get_time()
            return [ttl_feature]

        def get_local_device_features(idx):
            queue_length = len(self.dynetwork._network.nodes[idx]["sending_queue"])
            return np.array([queue_length])

        def get_neighborhood_device_features():
            avialable_neighbors = list(np.squeeze(np.argwhere(self.neighborhood_vec[src] != 0)))
            neighbors_features = []
            for neighbor in avialable_neighbors:
                neighbor_features = np.array(get_local_device_features(neighbor.item()))
                neighbors_features.append(neighbor_features)

            neighbors_features = np.array(neighbors_features)
            aggregated_features = np.concatenate((neighbors_features.max(axis=0), neighbors_features.min(axis=0), neighbors_features.mean(axis=0)))
            return aggregated_features

        pkt = self.dynetwork._packets.packetList[self.packet]
        src, dest = pkt.get_curPos(), pkt.get_endPos()
        if src == dest:
            return torch.zeros((1, self.state_dim)), src, dest
        state = torch.unsqueeze(torch.concat((self.neighborhood_vec[src], self.srcOneHot[src], self.user_association_rule_one_hot[dest-self.dest_offset])), dim=0)
        pkt_feat = get_packet_features()
        local_feat = get_local_device_features(src)
        neighborhood_feat = get_neighborhood_device_features()
        # local_features = np.concatenate((pkt_feat, local_feat))
        local_features = np.concatenate((pkt_feat, local_feat, neighborhood_feat))
        state = torch.unsqueeze(torch.concat((torch.squeeze(state), torch.Tensor(local_features))), dim=0)
        return state, src, dest
