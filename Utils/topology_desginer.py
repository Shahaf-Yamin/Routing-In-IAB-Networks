import numpy as np
import networkx as nx
import logging
import matplotlib.pyplot as plt
from Environment.TopologyEnv import IABNetworkNode, UENetworkNode, NetworkTopologyEnv, calculate_path_loss
import copy
from Environment.Buffer import PriorityQueue

logger = logging.getLogger("logger")
plt.close('all')

def plot_network_architecture(iab_list, ue_list, path, edges, graph):
    fig, ax = plt.subplots()
    x = iab_list[0].x
    y = iab_list[0].y
    donor = ax.scatter(x, y, s=200, zorder=1, color='r')
    ax.annotate('D', xy=(x, y), xytext=(5, 5), textcoords='offset points')

    '''
    IAB Nodes
    '''
    x = []
    y = []
    iab_names = np.arange(1, len(iab_list))
    for idx, iab in enumerate(iab_list[1:]):
        x.append(iab.x)
        y.append(iab.y)

    x = np.array(x)
    y = np.array(y)
    size = [100 for elem in iab_list[1:]]

    # Plot a circle of the BS/Relay
    nodes = ax.scatter(x, y, s=np.array(size),zorder=1,color='b')
    for x, y, name in zip(x, y, iab_names):
        # Write the BS/Relay Name above the corresponding circle
        ax.annotate(str(name), xy=(x, y), xytext=(5, 5), textcoords='offset points')

    '''
    User Equipment 
    '''
    x = []
    y = []
    ue_names = np.arange(0,len(ue_list))
    for idx, ue in enumerate(ue_list):
        x.append(ue.x)
        y.append(ue.y)

    x = np.array(x)
    y = np.array(y)
    size = [10 for elem in ue_list]

    # Plot a circle of the BS/Relay
    ues = ax.scatter(x, y, s=np.array(size), zorder=1, color='g')
    for x, y, name in zip(x, y, ue_names):
        # Write the BS/Relay Name above the corresponding circle
        ax.annotate(str(name), xy=(x, y), xytext=(5, 5), textcoords='offset points')

    nodes_legned = plt.legend((donor, nodes, ues),
               ('Donor', 'Node', 'UE'),
               scatterpoints=1,
               loc='best',
               ncol=2,
               fontsize=8)

    plt.grid()
    NUM_BASESTATIONS = len(iab_list)
    for TX_idx, RX_idx in edges:
        if TX_idx >= NUM_BASESTATIONS:
            continue
        TX = iab_list[TX_idx]
        if RX_idx >= NUM_BASESTATIONS:
            RX = ue_list[RX_idx - NUM_BASESTATIONS]
            access = True
        else:
            RX = iab_list[RX_idx]
            access = False

        if access:
            x = np.array([TX.x, RX.x])
            y = np.array([TX.y, RX.y])
            access = ax.plot(x, y, zorder=1, color='royalblue', linestyle='dashdot', alpha=0.2, label='access')
        else:
            backhaul = ax.annotate("", xy=(RX.x, RX.y), xytext=(TX.x, TX.y), arrowprops=dict(arrowstyle="->"), color='black', label='backhaul')
            # backhaul = ax.arrow(x, y, dx, dy,head_width=200, head_length=50, fc='k', ec='k', color='black', linestyle='dotted')
            x = np.array([TX.x, RX.x])
            y = np.array([TX.y, RX.y])
            backhaul = ax.plot(x, y, zorder=3, color='black')

    lines = ax.get_lines()
    legend1 = plt.legend([lines[i] for i in [0, -1]], ["Backhaul", "Access"], loc=4)
    ax.add_artist(legend1)
    ax.add_artist(nodes_legned)
    plt.savefig(path + '/network_architecture_pyplot.png', dpi=1024)

def EstablisheRandomNetworkConnections(env, Graph, verbose=False):
    def generateTopology(env, graph,scenraio:str, done:bool):
        def establish_connection(selectedNode, selectedParent):
            # In case that we have multiple choices for our children nodes
            if selectedNode is None:
                selectedNode = np.random.choice(possibleNodesForConnection)
            if verbose:
                print(f'selectedNode - {selectedNode}')
            if selectedParent is None:
                # The chosen Node Chooses Greedly based on the Available Parents and his spectral Eff Mat
                selectedParent = possibleParents[np.argmax((spectralEffMat * notAdjacenyMatrix)[possibleParents, selectedNode])]
            if verbose:
                print(f'selectedParent - {selectedParent}')
            return selectedNode, selectedParent
        assert scenraio == 'BS' or scenraio == 'UE'
        cnt = 0
        while not done and cnt < 20000:
            cnt += 1
            # Get current Env state
            state = env.state()
            # Set those values to invalid
            selectedNode = None
            selectedParent = None
            singleParentFlag = False
            singleChildrenFlag = False
            spectralEffMat, ParentsHist, ChildrenHist = state[0], state[1], state[2]
            if verbose:
                print(f'Parents histogram - {ParentsHist} (How many parents each node has)')
                print(f'Children histogram - {ChildrenHist} (How many childrens each node has)')
            # The 1 is for the Donor
            # Check how many parents each node has and extract which Base station is capable of being an IAB children
            if scenraio == 'BS':
                possibleNodesForConnection = np.squeeze(np.argwhere(ParentsHist[:NUM_BS] < availableParents))
                NodesNumParents = np.squeeze(ParentsHist[:NUM_BS][ParentsHist[:NUM_BS] < availableParents])
            else:
                possibleNodesForConnection = NUM_BS + np.squeeze(np.argwhere(ParentsHist[NUM_BS:] < avaliableUeParents))
                NodesNumParents = np.squeeze(ParentsHist[NUM_BS:][ParentsHist[NUM_BS:] < avaliableUeParents])
            if (NodesNumParents == 0).any():
                # In case that we have nodes without parents we would like to choose them first.
                possibleNodesForConnection = possibleNodesForConnection[NodesNumParents == 0]
            if verbose:
                print(f'possibleNodesForConnection - {possibleNodesForConnection}')
            # Check which Base station is capable of being an IAB children
            tempPossibleParents = np.squeeze(np.argwhere(np.logical_and((ChildrenHist < avaliableChildrens), np.logical_or(ParentsHist[:NUM_BS] > 0, donor_mask))))
            possibleParents = []
            for parent in tempPossibleParents.ravel():
                if nx.has_path(graph, 0, parent):
                    possibleParents.append(parent)
            possibleParents = np.array(possibleParents)
            if verbose:
                print(f'possibleParents - {possibleParents}')
            try:
                # Check if we've already finished to generate our Topology
                if possibleNodesForConnection.shape[0] == 0 or possibleParents.shape[0] == 0:
                    done = True
                    continue
            except IndexError:
                # Handle some edge cases of a single unit per children or father.
                if isinstance(possibleNodesForConnection, np.int64):
                    selectedNode = possibleNodesForConnection
                    singleChildrenFlag = True
                elif isinstance(possibleNodesForConnection, np.ndarray) and possibleNodesForConnection.shape == ():
                    selectedNode = possibleNodesForConnection.item()
                    singleChildrenFlag = True
                if isinstance(possibleParents, np.int64):
                    selectedParent = possibleParents
                    singleParentFlag = True
                if isinstance(possibleParents, np.ndarray) and possibleParents.shape == ():
                    selectedParent = possibleParents.item()
                    singleParentFlag = True
            trial = None
            for trial in range(100):
                selectedNode, selectedParent = establish_connection(selectedNode, selectedParent)
                if notAdjacenyMatrix[selectedParent, selectedNode] == 0:
                    if not singleParentFlag:
                        selectedParent = None
                    if not singleChildrenFlag:
                        selectedNode = None
                else:
                    break
            if trial == 99:
                break
            graph.add_edge(selectedParent, selectedNode)
            # if not nx.is_directed_acyclic_graph(graph):
            #     graph.remove_edge(selectedParent, selectedNode)
            #     continue
            # Set the decision in the not adjaceny matrix
            notAdjacenyMatrix[selectedParent, selectedNode] = 0
            if scenraio == 'BS':
                # if selectedParent != 0:
                notAdjacenyMatrix[selectedNode, selectedParent] = 0
                graph.add_edge(selectedNode, selectedParent)
            reward = env.step(selectedParent, selectedNode)

    availableParents, avaliableChildrens = env.avaliableParents, env.avaliableChildrens
    done = False
    NUM_BS = env.config.NETWORK["number Basestation"]
    NUM_UE = env.config.NETWORK["number user"]
    notAdjacenyMatrix = np.ones((NUM_BS, NUM_BS + NUM_UE)) - np.column_stack((np.eye(NUM_BS), np.zeros((NUM_BS, NUM_UE))))
    donor_mask = np.full(NUM_BS, False)
    donor_mask[0] = True
    # Set Up connections between Base stations
    generateTopology(env, Graph, scenraio='BS', done=done)
    # Set Up connections between Base stations and user equipments
    done = False
    avaliableChildrens += env.avaliableUEChildrens
    avaliableUeParents = env.avaliableUeParents
    generateTopology(env, Graph, scenraio='UE', done=done)

    return sum(env.calculateGraphScore()), env

def generateIABNetwork(config):
    env = NetworkTopologyEnv(config)
    Graph = nx.OrderedDiGraph()
    IABNodeIndices = [idx for idx in range(config.NETWORK['number Basestation'])]
    '''
    Generate position dictionary
    '''
    positions = {}
    for idx in range(config.NETWORK['number Basestation']):
        positions[idx] = np.array(env.IAB[idx])

    for idx in range(config.NETWORK['number user']):
        positions[idx + config.NETWORK['number Basestation']] = np.array(env.UE[idx])

    Graph.add_nodes_from(IABNodeIndices)
    '''IAB Node attributes'''
    nx.set_node_attributes(Graph, copy.deepcopy(config["NETWORK"]["sending capacity"]), 'max_send_capacity')
    nx.set_node_attributes(Graph, 0, 'max_queue_len')
    nx.set_node_attributes(Graph, 0, 'avg_q_len_array')
    nx.set_node_attributes(Graph, 0, 'growth')
    UENodeIndices = [idx for idx in range(config.NETWORK['number Basestation'], config.NETWORK['number Basestation'] + config.NETWORK['number user'])]
    Graph.add_nodes_from(UENodeIndices)

    ''' Add Queues for each network Node '''
    receiving_queue_dict, sending_queue_dict, locations = {}, {}, {}
    for i in range(len(IABNodeIndices) + len(UENodeIndices)):
        temp = {'receiving_queue': []}
        temp2 = {'sending_queue': PriorityQueue(config["NETWORK"]['holding capacity'], is_fifo=config["NETWORK"]["is_fifo"])}
        receiving_queue_dict.update({i: temp})
        sending_queue_dict.update({i: temp2})
    del temp, temp2
    nx.set_node_attributes(Graph, receiving_queue_dict)
    nx.set_node_attributes(Graph, sending_queue_dict)
    nx.set_node_attributes(Graph, copy.deepcopy(config["NETWORK"]["holding capacity"]), 'max_receive_capacity')
    nx.set_node_attributes(Graph, copy.deepcopy(positions), 'location')

    score, env = EstablisheRandomNetworkConnections(env, Graph)
    edges = env.edges


    ''' Add current wireless connections '''
    Graph.add_edges_from(edges)
    minNumHops = []
    ''' Visualize the statistics of the chosen topology '''
    number_of_users_histogram = np.zeros((config.NETWORK['number Basestation']))
    number_of_paths_per_users = np.zeros((config.NETWORK['number user']))
    for tx in range(0, config.NETWORK['number Basestation']):
        for node in range(config.NETWORK['number Basestation'], config.NETWORK['number user'] + config.NETWORK['number Basestation']):
            try:
                pathLen = nx.shortest_path_length(Graph, source=tx, target=node)
                number_of_users_histogram[tx] += 1
                number_of_paths_per_users[node - config.NETWORK['number Basestation']] += 1
            except nx.NetworkXNoPath:
                continue
    plt.figure()
    plt.xlabel('User Index')
    plt.ylabel('Number of Connected Base-stations')
    plt.bar(range(0, config.NETWORK['number user']), number_of_paths_per_users)
    plt.grid()
    plt.savefig(config.result_dir+'/user_avilable_paths_bar_plot.png', dpi=512)
    plt.figure()
    plt.xlabel('Bs Index')
    plt.ylabel('Number of Associated User Equipments')
    plt.bar(range(0, config.NETWORK['number Basestation']), number_of_users_histogram)
    plt.grid()
    plt.savefig(config.result_dir + '/user_associated_ues_per_bs_bar_plot.png', dpi=512)
    ''' Verify that the donor is capable reach any node '''
    for node in range(1, config.NETWORK['number Basestation'] + config.NETWORK['number user']):
        while True:
            try:
                pathLen = nx.shortest_path_length(Graph, source=0, target=node)
                minNumHops.append(pathLen)
                break
            except nx.NetworkXNoPath:
                raise Exception('Invalid Topology')

    print(f"Average Number of hops from Donor {np.mean(minNumHops)}")
    print(f"Maximal Number of hops from Donor {np.max(minNumHops)}")
    print(f"Minimal Number of hops from Donor {np.min(minNumHops)}")
    logger.info(f"Average Number of hops from Donor {np.mean(minNumHops)}")
    logger.info(f"Maximal Number of hops from Donor {np.max(minNumHops)}")
    logger.info(f"Minimal Number of hops from Donor {np.min(minNumHops)}")


    '''
    Plot the network architecture's topology.
    '''
    plot_network_architecture(env.IAB, env.UE, path=config.result_dir, edges=list(Graph.edges), graph=Graph)

    return Graph, positions

def generate_iab_topology(config):
    # Randomize locations
    x_coords = np.random.choice(config.NETWORK['max_x'], size=config.NETWORK['number Basestation'], replace=False)
    y_coords = np.random.choice(config.NETWORK['max_y'], size=config.NETWORK['number Basestation'], replace=False)

    # Generate node temporal classes
    donor_location = IABNetworkNode(0, 0, 20)
    nodes = [donor_location]
    for cnt, iab_index in enumerate(range(config.NETWORK['number Basestation']-1)):
        nodes.append(IABNetworkNode(x_coords[cnt], y_coords[cnt], 20))

    '''
    Calculate path loss matrix
    '''
    path_loss_matrix = np.zeros((config.NETWORK['number Basestation'], config.NETWORK['number Basestation']))
    for tx_node in range(config.NETWORK['number Basestation']):
        for rx_node in range(config.NETWORK['number Basestation']):
            tx = nodes[tx_node]
            rx = nodes[rx_node]
            if rx == tx:
                path_loss_matrix[tx_node, rx_node] = np.inf
                continue
            distance = tx - rx
            path_loss_matrix[tx_node, rx_node] = calculate_path_loss(distance=distance, frequency=28 * (10 ** 9),
                                                                     tx_antenna_gain_db=tx.TX, rx_antenna_gain_db=15)

    '''
    Generate connections in a greedy fashion
    '''
    nodeIndices = np.random.permutation(np.arange(1, config.NETWORK['number Basestation']))
    edges = []
    sampledParentsList = []
    for node in nodeIndices:
        Parents = np.random.randint(1, 4)
        sampledParentsList.append(Parents)
        indices = path_loss_matrix[:, node].argsort()[:Parents]
        for idx in indices:
            edges.append((idx, node))
            edges.append((node, idx))

    Graph = nx.OrderedDiGraph()
    Graph.add_nodes_from([i for i in range(config.NETWORK['number Basestation'])])
    Graph.add_edges_from(edges)
    minNumHops = []

    '''
    Verify that the donor is capable reach any node
    '''
    for node in nodeIndices:
        while True:
            try:
                pathLen = nx.shortest_path_length(Graph, source=0, target=node)
                minNumHops.append(pathLen)
                break
            except nx.NetworkXNoPath:
                numParents = sampledParentsList[node]
                index = path_loss_matrix[:, node].argsort()[numParents]
                sampledParentsList[node] += 1
                Graph.add_edges_from([(index, node), (node, index)])

    print(f"Average Number of hops from Donor {np.mean(minNumHops)}")
    logger.info(f"Average Number of hops from Donor {np.mean(minNumHops)}")
    '''
    Generate position dictionary
    '''
    positions = {}
    for idx in range(config.NETWORK['number Basestation']):
        positions[idx] = np.array(nodes[idx])

    plot_network_architecture(nodes, path=config.result_dir, edges=edges, graph=Graph)
    return Graph, positions

def generate_grid_topology(size):
    root = np.sqrt(size)
    assert int(root + 0.5) ** 2 == size
    edges = []
    root = int(root)
    for row in range(root):
        for col in range(root):
            if row != 0:
                # from above
                edges.append((row * root + col, (row-1) * root + col))
            if col != (root - 1):
                # from the right
                edges.append((row*root+col, row*root+col + 1))
            if row != (root - 1):
                # from below
                edges.append((row * root + col, (row + 1) * root + col))
            if col != 0:
                # from the left
                edges.append((row * root + col, row * root + col - 1))
    Graph = nx.OrderedDiGraph()
    Graph.add_nodes_from([i for i in range(size)])
    Graph.add_edges_from(edges)
    return Graph

def generate_att_topolgy():
    Graph = nx.OrderedDiGraph()
    Graph.add_nodes_from([i for i in range(25)])
    edges = []
    edges.append((0, 3))
    edges.append((1, 6))
    edges.append((2, 3))
    edges.append((2, 23))
    edges.append((3, 6))
    edges.append((3, 15))
    edges.append((4, 6))
    edges.append((5, 9))
    edges.append((5, 6))
    edges.append((6, 7))
    edges.append((7, 19))
    edges.append((8, 9))
    edges.append((9, 10))
    edges.append((9, 12))
    edges.append((9, 19))
    edges.append((9, 24))
    edges.append((10, 11))
    edges.append((10, 12))
    edges.append((10, 13))
    edges.append((10, 14))
    edges.append((14, 17))
    edges.append((15, 16))
    edges.append((15, 23))
    edges.append((17, 18))
    edges.append((17, 19))
    edges.append((19, 20))
    edges.append((19, 21))
    edges.append((21, 22))
    edges.append((22, 23))
    edges.append((23, 24))
    Graph.add_edges_from(edges)
    return Graph
