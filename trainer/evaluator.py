''' Incremental-Classifier Learning 
 Authors : Khurram Javed, Muhammad Talha Paracha
 Maintainer : Khurram Javed
 Lab : TUKL-SEECS R&D Lab
 Email : 14besekjaved@seecs.edu.pk '''
import copy
import logging

import numpy as np
from Agents import Agent
from tqdm import tqdm
import torch
from Environment.UpdateEdges import NodeFailure, NodeRestoreFailure

logger = logging.getLogger('logger')


class EvaluatorFactory():
    '''
    This class is used to get different versions of evaluators
    '''
    def __init__(self):
        pass

class RouterEvaluator(EvaluatorFactory):
    '''
    Evaluator class for softmax classification 
    '''
    def __init__(self, stat_collector, env, setting, writer, update_freq):
        super().__init__()
        self.arrival_rate_load = np.concatenate((np.arange(setting["Simulation"]["test_network_load_min"], setting["Simulation"]["test_network_load_max"],
                                                setting["Simulation"]["test_network_load_step_size"]), np.arange(setting["Simulation"]["test_network_load_max"] - setting["Simulation"]["test_network_load_step_size"], setting["Simulation"]["test_network_load_min"] - setting["Simulation"]["test_network_load_step_size"],
                                                - setting["Simulation"]["test_network_load_step_size"])))

        self.stat_collector = stat_collector
        self.env = env
        self.name = setting["NETWORK"]["Environment"]
        self.trials = setting["Simulation"]["test_trials_per_load"]
        self.time_steps = setting["Simulation"]["test_allowed_time_step_per_episode"]
        self.reward_func = setting["AGENT"]["rewardfunction"]
        self.writer = writer
        self.trials_counter = {key: 0 for key in set(self.arrival_rate_load)}
        self.training_freq = update_freq


    def evaluate(self, agent):
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            for curLoad in self.arrival_rate_load:
                print("---------- Testing Load of ", curLoad, " ----------")
                logger.info("---------- Testing Load of ", curLoad, " ----------")
                self.stat_collector.init_new_load(curLoad)
                self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=curLoad)

                ''' iterate each time step try to finish routing within time_steps '''
                for t in tqdm(range(self.time_steps)):
                    self.env.updateWhole(agent)
                print(f'{self.name} Routing Deliveries - [{curLoad}][{currTrial}] = {self.env.dynetwork._deliveries}')
                ''' STATS MEASURES '''
                self.stat_collector.add_load_trial_results(self.env, self.trials_counter[curLoad], self.name)
                self.trials_counter[curLoad] += 1
            self.saveStatsToTensorBoard(curLoad)

    def evaluateChangingLoad(self, agent):
        StartHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.6)]
        StopHighLoad = [int(self.time_steps * 0.4), int(self.time_steps * 0.8)]
        LowLoad = self.arrival_rate_load[2]
        HighLoad = self.arrival_rate_load[7]
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=LowLoad)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                self.env.updateWhole(agent)
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)
                # Modify the current Load if necessary
                if t in StartHighLoad:
                    self.env.dynetwork._lambda_load = HighLoad
                if t in StopHighLoad:
                    self.env.dynetwork._lambda_load = LowLoad

            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def evaluateNodeFailure(self, agent):
        # StartNodeFailure = [int(self.time_steps * 0.4)]
        StartNodeFailure = [int(self.time_steps * 0.4)]
        StopNodeFailure = []
        StopNodeFailure = [int(self.time_steps * 0.6)]
        links = []
        probs = None
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps
        nodeIdx = None
        for currTrial in range(self.trials):
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                self.env.updateWhole(agent)
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)
                # Modify the current Load if necessary
                if t in StartNodeFailure:
                    # Protection from the case of 2 consecutive trials of node's failure.
                    assert nodeIdx is None
                    # Sample uniformly a node from of the network's base-stations.
                    nodeIdx = 1 + np.random.choice(self.env.nnodes-1)
                    # Mark that this node has a failure.
                    links, probs = NodeFailure(self.env.dynetwork, nodeIdx)
                if t in StopNodeFailure:
                    NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                    nodeIdx = None
            if not StopNodeFailure:
                NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                nodeIdx = None
            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def saveStatsToTensorBoard(self, Load):
        # self.writer.add_scalar('Average Delay', np.mean(self.stat_collector.stats["Average Delay Time"][self.name][Load]), Load)
        # self.writer.add_scalar('Packet Drop Percentile', np.mean(self.stat_collector.stats["Packet Drop Percentile"][self.name][Load]), Load)
        pass

class TabularRLRouterEvaluator(RouterEvaluator):
    def __init__(self, stat_collector, env, setting, writer, update_freq):
        super().__init__(stat_collector, env, setting, writer, update_freq)

    def evaluate(self, agent):
        try:
            agent.config['epsilon'] = min(0.01, agent.config['epsilon'])
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            for curLoad in self.arrival_rate_load:
                print("---------- Testing Load of ", curLoad, " ----------")
                logger.info("---------- Testing Load of ", curLoad, " ----------")
                self.stat_collector.init_new_load(curLoad)
                logger.info('Trial ' + str(currTrial))
                self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=curLoad)
                '''iterate each time step try to finish routing within time_steps'''
                for t in tqdm(range(self.time_steps)):
                    learning_transitions, rewards = self.env.updateWhole(agent)
                    '''key function that obtain action and update Q-table'''
                    if t % self.training_freq == 0:
                        for transition in learning_transitions:
                            state, action, reward, done, currNode_neighbor_list, NextNode_neighbor_neighbors_list = transition
                            ''' We pass None as our current node to avoid pushing into memory '''
                            td_error_temp = agent.learn(state, action, reward, done, currNode_neighbor_list, NextNode_neighbor_neighbors_list)

                print(f'{self.name} Routing Deliveries - [{curLoad}][{currTrial}] = {self.env.dynetwork._deliveries}')
                ''' STATS MEASURES '''
                self.stat_collector.add_load_trial_results(self.env, self.trials_counter[curLoad], self.name)
                self.trials_counter[curLoad] += 1
            self.saveStatsToTensorBoard(curLoad)

    def evaluateChangingLoad(self, agent):
        LowLoad = self.arrival_rate_load[2]
        HighLoad = self.arrival_rate_load[7]
        StartHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.6)]
        StopHighLoad = [int(self.time_steps * 0.4), int(self.time_steps * 0.8)]
        # StartHighLoad = [int(self.time_steps * 0.1), int(self.time_steps * 0.3), int(self.time_steps * 0.5), int(self.time_steps * 0.7), int(self.time_steps * 0.9)]
        # StopHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.4), int(self.time_steps * 0.6), int(self.time_steps * 0.8)]
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        initialAgent = copy.deepcopy(agent)
        for currTrial in range(self.trials):
            agent = copy.deepcopy(initialAgent)
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=LowLoad)
            '''iterate each time step try to finish routing within time_steps'''
            for t in range(self.time_steps):
                learning_transitions, rewards = self.env.updateWhole(agent)
                '''key function that obtain action and update Q-table'''
                if t % self.training_freq == 0:
                    for transition in learning_transitions:
                        state, action, reward, done, currNode_neighbor_list, NextNode_neighbor_neighbors_list = transition
                        ''' We pass None as our current node to avoid pushing into memory '''
                        td_error_temp = agent.learn(state, action, reward, done, currNode_neighbor_list,
                                                    NextNode_neighbor_neighbors_list)
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)

                # Modify the current Load if necessary
                if t in StartHighLoad:
                    self.env.dynetwork._lambda_load = HighLoad
                if t in StopHighLoad:
                    self.env.dynetwork._lambda_load = LowLoad

            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def evaluateNodeFailure(self, agent):
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps
        try:
            agent.config['epsilon'] = 0.0
            agent.config['decay_rate'] = 1
        except:
            pass
        StartNodeFailure = [int(self.time_steps * 0.4)]
        StopNodeFailure = [int(self.time_steps * 0.6)]
        links = []
        probs = None
        nodeIdx = None
        initialAgent = copy.deepcopy(agent)
        env = copy.deepcopy(self.env)
        for currTrial in range(self.trials):
            agent = copy.deepcopy(initialAgent)
            self.env = copy.deepcopy(env)
            # self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                learning_transitions, rewards = self.env.updateWhole(agent)
                '''key function that obtain action and update Q-table'''
                if t % self.training_freq == 0:
                    for transition in learning_transitions:
                        state, action, reward, done, currNode_neighbor_list, NextNode_neighbor_neighbors_list = transition
                        ''' We pass None as our current node to avoid pushing into memory '''
                        td_error_temp = agent.learn(state, action, reward, done, currNode_neighbor_list, NextNode_neighbor_neighbors_list)
                    agent.update_epsilon()
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)

                # Modify the current Load if necessary
                if t in StartNodeFailure:
                    # Protection from the case of 2 consecutive trials of node's failure.
                    assert nodeIdx is None
                    # Sample uniformly a node from of the network's base-stations.
                    nodeIdx = 1 + np.random.choice(self.env.nnodes - 1)
                    # Mark that this node has a failure.
                    links, probs = NodeFailure(self.env.dynetwork, nodeIdx)
                    print(nodeIdx)
                if t in StopNodeFailure:
                    NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                    nodeIdx = None
            if not StopNodeFailure:
                NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                nodeIdx = None
            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

class A2cRouterEvaluator(RouterEvaluator):
    '''
    Evaluator class for softmax classification
    '''
    def __init__(self, stat_collector, env, setting, writer, update_freq):
        super().__init__(stat_collector, env, setting, writer, update_freq)


    def evaluate(self, agent):
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            for curLoad in self.arrival_rate_load:
                print("---------- Testing Load of ", curLoad, " ----------")
                logger.info("---------- Testing Load of ", curLoad, " ----------")
                self.stat_collector.init_new_load(curLoad)

                self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=curLoad)

                ''' iterate each time step try to finish routing within time_steps '''
                for t in tqdm(range(self.time_steps)):
                    learning_transitions = self.env.updateWhole(agent)
                    if t % self.training_freq == 0:
                        loss_actor, loss_critic, rewards = agent.learn(learning_transitions)
                    else:
                        loss_actor = 0
                        loss_critic = 0
                        if len(learning_transitions) != 0:
                            rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                        else:
                            rewards = 0

                print(f'{self.name} Routing Deliveries - [{curLoad}][{currTrial}] = {self.env.dynetwork._deliveries}')
                ''' STATS MEASURES '''
                self.stat_collector.add_load_trial_results(self.env, self.trials_counter[curLoad], self.name)
                self.trials_counter[curLoad] += 1
            self.saveStatsToTensorBoard(curLoad)

    def evaluateChangingLoad(self, agent):
        # StartHighLoad = [int(self.time_steps * 0.1), int(self.time_steps * 0.3), int(self.time_steps * 0.5), int(self.time_steps * 0.7), int(self.time_steps * 0.9)]
        # StopHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.4), int(self.time_steps * 0.6), int(self.time_steps * 0.8)]
        StartHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.6)]
        StopHighLoad = [int(self.time_steps * 0.4), int(self.time_steps * 0.8)]
        LowLoad = self.arrival_rate_load[2]
        HighLoad = self.arrival_rate_load[7]
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=LowLoad)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                learning_transitions = self.env.updateWhole(agent)
                if t % self.training_freq == 0:
                    loss_actor, loss_critic, rewards = agent.learn(learning_transitions)
                else:
                    loss_actor = 0
                    loss_critic = 0
                    if len(learning_transitions) != 0:
                        rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                    else:
                        rewards = 0
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)
                # Modify the current Load if necessary
                if t in StartHighLoad:
                    self.env.dynetwork._lambda_load = HighLoad
                if t in StopHighLoad:
                    self.env.dynetwork._lambda_load = LowLoad

            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def evaluateNodeFailure(self, agent):
        StartNodeFailure = [int(self.time_steps * 0.4)]
        StopNodeFailure = []
        StopNodeFailure = [int(self.time_steps * 0.6)]
        links = []
        probs = None
        nodeIdx = None
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                learning_transitions = self.env.updateWhole(agent)
                if t % self.training_freq == 0:
                    loss_actor, loss_critic, rewards = agent.learn(learning_transitions)
                else:
                    loss_actor = 0
                    loss_critic = 0
                    if len(learning_transitions) != 0:
                        rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                    else:
                        rewards = 0
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)
                # Modify the current Load if necessary
                if t in StartNodeFailure:
                    # Protection from the case of 2 consecutive trials of node's failure.
                    assert nodeIdx is None
                    # Sample uniformly a node from of the network's base-stations.
                    nodeIdx = 1 + np.random.choice(self.env.nnodes - 1)
                    # Mark that this node has a failure.
                    links, probs = NodeFailure(self.env.dynetwork, nodeIdx)
                if t in StopNodeFailure:
                    NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                    nodeIdx = None
            if not StopNodeFailure:
                NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                nodeIdx = None
            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def saveStatsToTensorBoard(self, Load):
        # self.writer.add_scalar('Average Delay', np.mean(self.stat_collector.stats["Average Delay Time"][self.name][Load]), Load)
        # self.writer.add_scalar('Packet Drop Percentile', np.mean(self.stat_collector.stats["Packet Drop Percentile"][self.name][Load]), Load)
        pass

class RLFedratedA2CEvaluator(RouterEvaluator):
    '''
    Evaluator class for softmax classification
    '''
    def __init__(self, stat_collector, env, setting, writer, update_freq):
        super().__init__(stat_collector, env, setting, writer, update_freq)
        self.update_federated_freq = setting["Simulation"]["num_time_step_to_update_target_network"]

    def evaluate(self, agent):
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            for curLoad in self.arrival_rate_load:
                print("---------- Testing Load of ", curLoad, " ----------")
                logger.info("---------- Testing Load of ", curLoad, " ----------")
                self.stat_collector.init_new_load(curLoad)

                self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=curLoad)

                ''' iterate each time step try to finish routing within time_steps '''
                for t in tqdm(range(self.time_steps)):
                    learning_transitions = self.env.updateWhole(agent)
                    if t % self.training_freq == 0:
                        loss_actor, loss_critic, rewards = agent.learn(learning_transitions)
                    else:
                        loss_actor = 0
                        loss_critic = 0
                        if len(learning_transitions) != 0:
                            rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                        else:
                            rewards = 0
                    if t % self.update_federated_freq == 0:
                        agent.update_central_controller_weights()

                print(f'{self.name} Routing Deliveries - [{curLoad}][{currTrial}] = {self.env.dynetwork._deliveries}')
                ''' STATS MEASURES '''
                self.stat_collector.add_load_trial_results(self.env, self.trials_counter[curLoad], self.name)
                self.trials_counter[curLoad] += 1
            self.saveStatsToTensorBoard(curLoad)

    def evaluateChangingLoad(self, agent):
        # StartHighLoad = [int(self.time_steps * 0.1), int(self.time_steps * 0.3), int(self.time_steps * 0.5), int(self.time_steps * 0.7), int(self.time_steps * 0.9)]
        # StopHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.4), int(self.time_steps * 0.6), int(self.time_steps * 0.8)]
        LowLoad = self.arrival_rate_load[2]
        HighLoad = self.arrival_rate_load[7]
        StartHighLoad = [int(self.time_steps * 0.2), int(self.time_steps * 0.6)]
        StopHighLoad = [int(self.time_steps * 0.4), int(self.time_steps * 0.8)]
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=LowLoad)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                learning_transitions = self.env.updateWhole(agent)
                if t % self.training_freq == 0:
                    loss_actor, loss_critic, rewards = agent.learn(learning_transitions)
                else:
                    loss_actor = 0
                    loss_critic = 0
                    if len(learning_transitions) != 0:
                        rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                    else:
                        rewards = 0
                if t % self.update_federated_freq == 0:
                    agent.update_central_controller_weights()
                # Modify the current Load if necessary
                if t in StartHighLoad:
                    self.env.dynetwork._lambda_load = HighLoad
                if t in StopHighLoad:
                    self.env.dynetwork._lambda_load = LowLoad
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)

            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def evaluateNodeFailure(self, agent):
        StartNodeFailure = [int(self.time_steps * 0.4)]
        StopNodeFailure = []
        StopNodeFailure = [int(self.time_steps * 0.6)]
        links = []
        probs = None
        nodeIdx = None
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps

        for currTrial in range(self.trials):
            self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func)
            '''iterate each time step try to finish routing within time_steps'''
            for t in tqdm(range(self.time_steps)):
                learning_transitions = self.env.updateWhole(agent)
                if t % self.training_freq == 0:
                    loss_actor, loss_critic, rewards = agent.learn(learning_transitions)
                else:
                    loss_actor = 0
                    loss_critic = 0
                    if len(learning_transitions) != 0:
                        rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                    else:
                        rewards = 0
                if t % self.update_federated_freq == 0:
                    agent.update_central_controller_weights()
                # Modify the current Load if necessary
                if t in StartNodeFailure:
                    # Protection from the case of 2 consecutive trials of node's failure.
                    assert nodeIdx is None
                    # Sample uniformly a node from of the network's base-stations.
                    nodeIdx = 1 + np.random.choice(self.env.nnodes - 1)
                    # Mark that this node has a failure.
                    links, probs = NodeFailure(self.env.dynetwork, nodeIdx)
                    actorPolicy = agent.actors[nodeIdx].policy_net.state_dict()
                    criticPolicy = agent.critics[nodeIdx].network.state_dict()
                if t in StopNodeFailure:
                    NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                    agent.actors[nodeIdx].policy_net.load_state_dict(actorPolicy)
                    agent.critics[nodeIdx].network.load_state_dict(criticPolicy)
                    nodeIdx = None
                ''' STATS MEASURES '''
                self.stat_collector.add_stats(self.env, self.name, currTrial, currTimeslot=t)

            if not StopNodeFailure:
                NodeRestoreFailure(self.env.dynetwork, links, probs, index=nodeIdx)
                nodeIdx = None
            print(f'{self.name} Routing Deliveries - [{currTrial}] = {self.env.dynetwork._deliveries}')

    def saveStatsToTensorBoard(self, Load):
        # self.writer.add_scalar('Average Delay', np.mean(self.stat_collector.stats["Average Delay Time"][self.name][Load]), Load)
        # self.writer.add_scalar('Packet Drop Percentile', np.mean(self.stat_collector.stats["Packet Drop Percentile"][self.name][Load]), Load)
        pass

class TopologyRouterEvaluator(EvaluatorFactory):
    '''
    Evaluator class for softmax classification
    '''
    def __init__(self, stat_collector, env, setting):
        super().__init__()
        self.arrival_rate_load = np.arange(setting["Simulation"]["test_network_load_min"], setting["Simulation"]["test_network_load_max"],
                                           setting["Simulation"]["test_network_load_step_size"])
        self.stat_collector = stat_collector
        self.env = env
        self.name = setting["NETWORK"]["Environment"]
        self.trials = setting["Simulation"]["test_trials_per_load"]
        self.time_steps = setting["Simulation"]["test_allowed_time_step_per_episode"]
        self.reward_func = setting["AGENT"]["rewardfunction"]

    def evaluate(self, agent):
        try:
            agent.config['epsilon'] = 0.0001
            agent.config['decay_rate'] = 1
        except:
            pass
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps
        for curLoad in self.arrival_rate_load:
            print("---------- Testing Load of ", curLoad, " ----------")
            self.stat_collector.init_new_load(curLoad)

            for currTrial in range(self.trials):
                self.env.reset_env(seed=currTrial, reset_stochastic_engine=False, rewardfun=self.reward_func, arrival_rate=curLoad, resetTopology=True)

                ''' iterate each time step try to finish routing within time_steps '''
                for t in tqdm(range(self.time_steps)):
                    self.env.updateWhole(agent)
                print(f'{self.name} Routing Deliveries - [{curLoad}][{currTrial}] = {self.env.dynetwork._deliveries}')
                ''' STATS MEASURES '''
                self.stat_collector.add_load_trial_results(self.env, currTrial, self.name)


