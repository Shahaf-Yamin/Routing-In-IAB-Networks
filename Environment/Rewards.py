'''Hybrid function that penalizes proportional to queue size'''
import networkx as nx

TARGET_QUEUE_LENGTH = 100

def reward1(env, next_step, dest_node, weight):
    """ we reward the packet for being sent to a node according to our current reward function """
    path_len = nx.shortest_path_length(env.dynetwork._network, next_step, dest_node)
    queue_size = len(env.dynetwork._network.nodes[next_step]['sending_queue']) + len(env.dynetwork._network.nodes[next_step]['receiving_queue'])
    return (- path_len - (queue_size + weight))

'''Hybrid function that penalizes only if queue size exceeds a threshold'''

def reward2(env, next_step, dest_node, weight):
    path_len = nx.shortest_path_length(env.dynetwork._network, next_step, dest_node)
    queue_size = len(env.dynetwork._network.nodes[next_step]['sending_queue']) + len(env.dynetwork._network.nodes[next_step]['receiving_queue'])
    emptying_size = weight * env.max_transmit
    if queue_size > emptying_size:
        fullness = (queue_size - emptying_size)
    else:
        fullness = 0
    return (- path_len - (fullness + weight))

'''Function that only takes into account path length'''

def reward3(env, next_step, dest_node, weight):
    path_len = nx.shortest_path_length(env.dynetwork._network, next_step, dest_node)
    return -(path_len)

'''Function similar to reward2 that does not use shortest path'''

def reward4(env, next_step, dest_node, weight):
    queue_size = len(env.dynetwork._network.nodes[next_step]['sending_queue']) + len(env.dynetwork._network.nodes[next_step]['receiving_queue'])
    emptying_size = weight * env.max_transmit
    if queue_size > emptying_size:
        fullness = (queue_size - emptying_size)
    else:
        fullness = 0
    return (-(fullness + weight))

'''Function that penalizes proportional to difference in queue size and an equilibrium queue size,
   as well as growth rate of the queue'''

def reward5(env, next_step, dest_node, weight):
    q = len(env.dynetwork._network.nodes[next_step]['sending_queue']) + len(env.dynetwork._network.nodes[next_step]['receiving_queue'])
    q_eq = 0.8 * env.max_queue
    w = 5
    growth = env.dynetwork._network.nodes[next_step]['growth']
    return (- (q + w * growth - q_eq) ** 2)

'''Function that penalizes for exceeding equilibrium queue size, as well as growth rate of the queue'''

def reward6(env, next_step, dest_node, weight):
    q = len(env.dynetwork._network.nodes[next_step]['sending_queue']) + len(env.dynetwork._network.nodes[next_step]['receiving_queue'])
    q_eq = 0.8 * env.max_queue
    excess = q - q_eq
    w = 5
    growth = env.dynetwork._network.nodes[next_step]['growth']
    if excess > 0:
        return -(excess + w * growth)
    else:
        return -(w * growth)

'''Reward function that is equal to original Q-routing algorithm'''
def reward7(env, next_step, dest_node, weight):
    queue_size = len(env.dynetwork._network.nodes[next_step]['sending_queue']) + len(env.dynetwork._network.nodes[next_step]['receiving_queue'])
    return - (queue_size + weight)

def reward8(env, next_step, dest_node, weight):
    return -weight

