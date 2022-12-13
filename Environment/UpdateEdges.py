import numpy as np
import random
import math
from copy import deepcopy
from Utils.topology_desginer import calculate_path_loss
''' Functions to handle edges in our network. '''

class UserMovement:
    def __init__(self, dyNetwork, numBs, numUsers, maximalNumUsers, maximalParents, maxGrid, speed):
        self.dyNetwork = dyNetwork
        self.numBs = numBs
        self.numUsers = numUsers
        self.maximalNumUsers = maximalNumUsers
        self.maximalParents = maximalParents
        self.speed = 1.0
        self.movementDirection = np.array([self.speed if ueIdx % 2 == 0 else -self.speed for ueIdx in range(self.numUsers)])
        self.maxGrid = maxGrid

    def Movement(self, location: tuple, ueIdx: int, locationDelta):
        tempNewLocation = location + self.movementDirection[ueIdx] * locationDelta
        newLocation = tuple(map(lambda x: x if x > 0 and x < self.maxGrid else max(min(x, self.maxGrid), 0), tempNewLocation))
        if (newLocation != tempNewLocation).any():
            # Apply Wall Bouncing in this case
            self.movementDirection[ueIdx] *= -1
        return newLocation

    def calculateLocationDelta(self, timeslot):
        phase = (math.pi / 125) * timeslot % (2 * math.pi)
        return np.array((75 * np.cos(phase), 75 * np.sin(phase)), dtype=np.int64)

    def distance(self, a: tuple, b: tuple):
        ''' Calculates the 2D distance between 'a' and 'b' '''
        return np.linalg.norm(np.array(a) - np.array(b), ord=2)

    def GetNumberAssociatedUsers(self, bs_idx):
        return sum(list(map(lambda x: 0 if x < self.numBs else 1, list(self.dyNetwork._network.neighbors(bs_idx)))))

    def applyMovement(self, timeslot, closedStations):
        # Calculate how many connected users each Base-station owns
        num_users = np.zeros((self.numBs))
        for bs in range(0, self.numBs):
            num_users[bs] = self.GetNumberAssociatedUsers(bs_idx=bs)

        locationDelta = self.calculateLocationDelta(timeslot)
        userChanges = {}
        # Iterate over the user-equipments
        for ue in range(self.numBs, self.numUsers + self.numBs):
            # Update location
            self.dyNetwork._network.nodes[ue]['location'] = self.Movement(self.dyNetwork._network.nodes[ue]['location'], ueIdx=ue - self.numBs, locationDelta=locationDelta)
            path_losses = np.zeros((self.numBs))
            is_avialable = np.zeros((self.numBs), dtype=np.bool)
            # Calculate connected Base-stations
            current_connections = np.array(list(self.dyNetwork._network.in_edges(ue)))[:, 0]
            is_avialable[current_connections] = True
            # Calculate Path-Losses w.r.t all other base-stations
            for bs in range(0, self.numBs):
                # Calculate the euclidean distance between those base-stations
                curr_dist = self.distance(self.dyNetwork._network.nodes[ue]['location'], self.dyNetwork._network.nodes[bs]['location'])
                # Calculate the path-loss between those base-stations
                path_losses[bs] = calculate_path_loss(distance=curr_dist, frequency=28 * (10 ** 9), tx_antenna_gain_db=10,rx_antenna_gain_db=0) if curr_dist != 0 else 0
                # Setup the indicator that represents if the base-station is willing to receive new-users
                is_avialable[bs] = True if num_users[bs] < self.maximalNumUsers or is_avialable[bs] else False
            # Set the flag of station that are closed to False since users can't establish a connection with them.
            is_avialable[closedStations == True] = False
            path_losses[~is_avialable] = np.infty
            # Follow the greedy decision w.r.t the path losses between the available base-stations
            new_parents = path_losses.argsort()[:self.maximalParents]
            # Save only the new parents
            new_parents = new_parents[np.isfinite(path_losses[new_parents])]
            # Get the new wireless connections that we would like to establish
            new_links = [link for link in new_parents if link not in current_connections]
            # Get the old wireless connections that we would like to disconnect
            disconnected_links = [connection for connection in current_connections if connection not in new_parents]
            # Update the number of users per base-station
            num_users[new_links] += 1
            num_users[disconnected_links] -= 1
            # Add the wireless connections to the list
            randomDelay = np.random.randint(10)
            initial_edge_state = dict([('edge_delay', randomDelay), ('edge_global_delay', randomDelay), ('edge_error_probability', 0.0), ('sine_state', 0.0), ('initial_weight', randomDelay), ('initial_per', 0.0)])
            self.dyNetwork._network.add_edges_from([(bs, ue, deepcopy(initial_edge_state)) for bs in new_links])
            # Remove previous wireless connections
            self.dyNetwork._network.remove_edges_from([(bs, ue, self.dyNetwork._network[bs][ue]) for bs in disconnected_links])
            userChanges[ue] = {'new-connections': new_links, 'disconnected-connections': disconnected_links}

        return userChanges

class DirectedUserMovement(UserMovement):
    def __init__(self, dyNetwork, numBs, numUsers, maximalNumUsers, maximalParents, maxGrid, speed):
        super().__init__(dyNetwork, numBs, numUsers, maximalNumUsers, maximalParents, maxGrid, speed)
        self.movementDirection *= speed / 10

    def calculateLocationDelta(self, timeslot):
        return np.array((1, 1), dtype=np.float64)

''' Users Movement '''
def UEMovement(dyNetwork, numBs, numUsers, maximalNumUsers, maximalParents, timeslot, closedStations):
    def distance(a: tuple, b: tuple):
        ''' Calculates the 2D distance between 'a' and 'b' '''
        return np.linalg.norm(np.array(a)-np.array(b), ord=2)

    def GetNumberAssociatedUsers(bs_idx):
        return sum(list(map(lambda x: 0 if x < numBs else 1, list(dyNetwork._network.neighbors(bs_idx)))))

    # Calculate how many connected users each Base-station owns
    num_users = np.zeros((numBs))
    for bs in range(0, numBs):
        num_users[bs] = GetNumberAssociatedUsers(bs_idx=bs)
    initial_edge_state = dict([('edge_delay', 1), ('edge_global_delay', 1), ('edge_error_probability', 0.0), ('sine_state', 0.0), ('initial_weight', 1), ('initial_per', 0.0)])
    phase = (math.pi / 125) * timeslot % (2*math.pi)
    userChanges = {}
    # Iterate over the user-equipments
    for ue in range(numBs, numUsers+numBs):
        # Update location
        dyNetwork._network.nodes[ue]['location'] += ((-1) ** ue) * np.array((75 * np.cos(phase), 75 * np.sin(phase)), dtype=np.int64)
        path_losses = np.zeros((numBs))
        is_avialable = np.zeros((numBs), dtype=np.bool)
        # Calculate connected Base-stations
        current_connections = np.array(list(dyNetwork._network.in_edges(ue)))[:, 0]
        is_avialable[current_connections] = True
        # Calculate Path-Losses w.r.t all other base-stations
        for bs in range(0, numBs):
            # Calculate the euclidean distance between those base-stations
            curr_dist = distance(dyNetwork._network.nodes[ue]['location'], dyNetwork._network.nodes[bs]['location'])
            # Calculate the path-loss between those base-stations
            path_losses[bs] = calculate_path_loss(distance=curr_dist, frequency=28 * (10 ** 9), tx_antenna_gain_db=10, rx_antenna_gain_db=0) if curr_dist != 0 else 0
            # Setup the indicator that represents if the base-station is willing to receive new-users
            is_avialable[bs] = True if num_users[bs] < maximalNumUsers or is_avialable[bs] else False
        # Set the flag of station that are closed to False since users can't establish a connection with them.
        is_avialable[closedStations == True] = False
        path_losses[~is_avialable] = np.infty
        # Follow the greedy decision w.r.t the path losses between the available base-stations
        new_parents = path_losses.argsort()[:maximalParents]
        # Save only the new parents
        new_parents = new_parents[np.isfinite(path_losses[new_parents])]
        # Get the new wireless connections that we would like to establish
        new_links = [link for link in new_parents if link not in current_connections]
        # Get the old wireless connections that we would like to disconnect
        disconnected_links = [connection for connection in current_connections if connection not in new_parents]
        # Update the number of users per base-station
        num_users[new_links] += 1
        num_users[disconnected_links] -= 1
        # Add the wireless connections to the list
        randomDelay = np.random.randint(4)
        initial_edge_state = dict([('edge_delay', randomDelay), ('edge_global_delay', randomDelay), ('edge_error_probability', 0.0), ('sine_state', 0.0), ('initial_weight', randomDelay), ('initial_per', 0.0)])
        dyNetwork._network.add_edges_from([(bs, ue, deepcopy(initial_edge_state)) for bs in new_links])
        # Remove previous wireless connections
        dyNetwork._network.remove_edges_from([(bs, ue, dyNetwork._network[bs][ue]) for bs in disconnected_links])
        userChanges[ue] = {'new-connections': new_links, 'disconnected-connections': disconnected_links}

    return userChanges

''' Randomly deletes some number of edges between min_edge_removal and max_edge_removal '''
def Delete(dyNetwork, min_edge_removal, max_edge_removal):
    edges = dyNetwork._network.edges()
    deletion_number = random.randint(min_edge_removal, min(max_edge_removal, len(edges) - 1))
    strip = random.sample(edges, k=deletion_number)
    temp = []
    for s_edge, e_edge in strip:
        temp.append((s_edge, e_edge, dyNetwork._network[s_edge][e_edge]))
    strip = temp
    dyNetwork._network.remove_edges_from(strip)
    dyNetwork._stripped_list.extend(strip)

''' Removes a node from the graph due to failure '''
def NodeFailure(dyNetwork, index):
    fallenInEdges = dyNetwork._network.in_edges(index)
    fallenOutEdges = dyNetwork._network.out_edges(index)
    temp = []
    for s_edge, e_edge in fallenInEdges:
        temp.append((s_edge, e_edge, dyNetwork._network[s_edge][e_edge]))
    for s_edge, e_edge in fallenOutEdges:
        temp.append((s_edge, e_edge, dyNetwork._network[s_edge][e_edge]))
    strip = temp
    dyNetwork._network.remove_edges_from(strip)
    # Save the previous generation probabilities.
    prevProb = np.copy(dyNetwork.probGenerationNodes)
    # Reset the generation probability for the fallen node.
    dyNetwork.probGenerationNodes[index-1] = 0
    # Normalize the remaining generation probabilities.
    dyNetwork.probGenerationNodes = dyNetwork.probGenerationNodes / sum(dyNetwork.probGenerationNodes)
    # Mark at the simulation that this node functionality is damaged at the moment
    dyNetwork.closed_stations[index] = True
    return strip, prevProb

''' Restores a node from the graph due to recovery '''
def NodeRestoreFailure(dyNetwork, links, prevProb, index):
    dyNetwork._network.add_edges_from(links)
    dyNetwork.probGenerationNodes = prevProb
    # Mark at the simulation that this node functionality is no longer damaged
    dyNetwork.closed_stations[index] = True
    return

''' Randomly restores some edges we have deleted '''
def Restore(dyNetwork, max_edge_removal):
    restore_number = min(max_edge_removal, len(dyNetwork._stripped_list))
    restore = random.sample(dyNetwork._stripped_list, k=restore_number)
    dyNetwork._network.add_edges_from(restore)
    for edge in restore:
        # Return those edges to the strip list
        dyNetwork._stripped_list.remove(edge)

''' Randomly change edge weights '''
def Random_Walk(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        try:
            changed = random.randint(-2, 2) + dyNetwork._network[s_edge][e_edge]['edge_delay']
            dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(changed, 1)
        except:
            print(s_edge, e_edge)
            
''' Change edge weights so that the edge weight changes will be roughly sinusoidal across the simulation '''
def Sinusoidal(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(1, int(dyNetwork._network[s_edge][e_edge]['initial_weight']* (1 + 0.5 * math.sin(dyNetwork._network[s_edge][e_edge]['sine_state']))))
        dyNetwork._network[s_edge][e_edge]['edge_global_delay'] = dyNetwork._network[s_edge][e_edge]['edge_delay']
        dyNetwork._network[s_edge][e_edge]['sine_state'] += math.pi/6

def SinusoidalGlobal(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        dyNetwork._network[s_edge][e_edge]['edge_delay'] = max(1, int(dyNetwork._network[s_edge][e_edge]['initial_weight'] * (1 + 0.5 * math.sin(dyNetwork._network[s_edge][e_edge]['sine_state']))))
        dyNetwork._network[s_edge][e_edge]['edge_global_delay'] = len(dyNetwork._network.nodes[e_edge]['sending_queue']) + len(dyNetwork._network.nodes[e_edge]['receiving_queue']) + dyNetwork._network[s_edge][e_edge]['edge_delay']
        dyNetwork._network[s_edge][e_edge]['sine_state'] += math.pi/6

def updateGlobalEdges(dyNetwork):
    for s_edge, e_edge in dyNetwork._network.edges():
        dyNetwork._network[s_edge][e_edge]['edge_global_delay'] = len(dyNetwork._network.nodes[e_edge]['sending_queue']) + len(dyNetwork._network.nodes[e_edge]['receiving_queue']) + dyNetwork._network[s_edge][e_edge]['edge_delay']

''' Not in use. If it were used the edge weight would be the average of the number of packets in each queue of the endpoints of the edge. '''
def Average(dyNetwork):
    for node1, node2 in dyNetwork._network.edges(data = False):
        tot_queue1 = dyNetwork._network.nodes[node1]['sending_queue']
        tot_queue2 = dyNetwork._network.nodes[node2]['sending_queue']
        avg = np.avg([tot_queue1, tot_queue2])
        dyNetwork._network[node1][node2]['edge_delay'] = avg
        del tot_queue1, tot_queue2