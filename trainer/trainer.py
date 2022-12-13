
from __future__ import print_function

import logging

from torch.autograd import Variable
# from tensorflow import summary
logger = logging.getLogger('logger')
import numpy as np
import copy
import networkx as nx
import torch
from tqdm import tqdm


class GenericTrainer:
    '''
    Base class for trainer; to implement a new training routine, inherit from this. 
    '''

    def __init__(self):
        pass

class RLTrainer(GenericTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq):
        super().__init__()
        self.time_steps = time_steps
        self.target_update = TARGET_UPDATE
        self.agent = agent
        self.stat_collector = stat_collector
        self.env = env
        # Set up simulation ending time for this scenario
        self.env.simulation_ending_time = self.time_steps
        self.name = name
        self.writer = writer
        self.experiment = experiment
        self.training_freq = update_freq
        self.arrival_rate_load = np.arange(self.env.setting["Simulation"]["test_network_load_min"], self.env.setting["Simulation"]["test_network_load_max"], 1)

    # def update_lr(self, epoch, schedule, gammas):
    #     for temp in range(0, len(schedule)):
    #         if schedule[temp] == epoch:
    #             for param_group in self.optimizer.param_groups:
    #                 self.current_lr = param_group['lr']
    #                 param_group['lr'] = self.current_lr * gammas[temp]
    #                 logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
    #                              self.current_lr * gammas[temp])
    #                 self.current_lr *= gammas[temp]

    def train(self, episode):
        pass

    def saveStatsToTensorBoard(self, rewards, episode, timestep, td_errors, delay):
        STEP = episode * self.time_steps + timestep
        # self.writer.add_scalar('Epsilon', self.agent.config["epsilon"], STEP)
        # self.writer.add_scalar('Reward', rewards, STEP)
        # self.writer.add_scalar('Avg_Loss', np.mean(td_errors), STEP)
        # self.writer.add_scalar('Max_Loss', np.max(td_errors), STEP)
        # self.writer.add_scalar('Min_Loss', np.min(td_errors), STEP)
        # self.experiment.log({'Avg Loss': np.mean(td_errors), 'step': STEP, 'episode': episode})
        # self.experiment.log({'Max Loss': np.max(td_errors), 'step': STEP, 'episode': episode})
        # self.experiment.log({'Min Loss': np.min(td_errors), 'step': STEP, 'episode': episode})
        # self.experiment.log({'Reward': rewards, 'step': STEP, 'episode': episode})
        # self.experiment.log({'Epsilon': self.agent.config["epsilon"], 'step': STEP, 'episode': episode})
        # self.experiment.log({'Delay': delay, 'step': STEP, 'episode': episode})

class RLTabularTrainer(RLTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq):
        super().__init__(time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq)

    # def update_lr(self, epoch, schedule, gammas):
    #     for temp in range(0, len(schedule)):
    #         if schedule[temp] == epoch:
    #             for param_group in self.optimizer.param_groups:
    #                 self.current_lr = param_group['lr']
    #                 param_group['lr'] = self.current_lr * gammas[temp]
    #                 logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
    #                              self.current_lr * gammas[temp])
    #                 self.current_lr *= gammas[temp]

    def train(self, episode):
        print("---------- Episode:", episode + 1, " ----------")

        step = []
        deliveries = []
        pbar = tqdm(range(self.time_steps))
        if episode != 0:
            self.env.dynetwork._lambda_load = self.arrival_rate_load[episode % self.arrival_rate_load.shape[0]]
        '''iterate each time step try to finish routing within time_steps'''
        for t in pbar:
            if t % 1000 == 0:
                logger.info(f'Timestep - {t}')
            '''key function that obtain action and update Q-table'''
            learning_transitions, rewards = self.env.updateWhole(self.agent)
            td_errors = []
            estimationVsReturnError = []
            self.env.updateDelayMapping()
            sizeInformationExchange = 0

            for transition in learning_transitions:
                state, action, reward, done , currNode_neighbor_list, NextNode_neighbor_neighbors_list = transition
                ''' We pass None as our current node to avoid pushing into memory '''
                if t % self.training_freq == 0:
                    td_error_temp, infoSize = self.agent.learn(state, action, reward, done, currNode_neighbor_list, NextNode_neighbor_neighbors_list)
                    sizeInformationExchange += infoSize
                    if td_error_temp is not None:
                        td_errors.append(td_error_temp)
                    # currPos, destination = state
                    # if type(currPos) == tuple:
                    #     state = currPos
                    #     currPos, destination = currPos
                    # estimationVsReturnError.append(-self.env.trip_delay_mapping[currPos][destination]-self.agent.q[state][action])
                    estimationVsReturnError.append(0)
                else:
                    td_errors = [0]
                    estimationVsReturnError = [0]

            pbar.set_postfix({'Reward': rewards, 'Epsilon': self.agent.config["epsilon"], 'Loss': np.mean(td_errors), 'Delay': sum(self.env.dynetwork._delivery_times[-200:]) / (len(self.env.dynetwork._delivery_times[-200:])+1e-7)})
            ''' Update the exploration rate for all agents '''
            self.agent.update_epsilon()
            '''store atributes for performance measures'''
            step.append(t)
            deliveries.append(copy.deepcopy(self.env.dynetwork._deliveries))

            ''' Save time-step trends to stat collector '''
            self.stat_collector.add_timestep_trends(self.env, self.name, td_errors, rewards, estimationVsReturnError, sizeInformationExchange)

            ''' Save time-step trends to tensorboard '''
            # self.saveStatsToTensorBoard(rewards, episode, timestep=t, td_errors=td_errors)

        print('Number of packets -> ', self.env.dynetwork._deliveries)
        '''Save all performance measures'''
        self.stat_collector.add_stats(self.env, self.name)
        print(self.env.calc_avg_delivery())
        self.env.reset_env()

class RLA2CTrainer(RLTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq):
        super().__init__(time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq)
    def train(self, episode):
        print("---------- Episode:", episode + 1, " ----------")
        step = []
        deliveries = []
        pbar = tqdm(range(self.time_steps))
        mean_td_error = 0
        '''iterate each time step try to finish routing within time_steps'''
        if episode != 0:
            self.env.dynetwork._lambda_load = self.arrival_rate_load[episode % self.arrival_rate_load.shape[0]]
        for t in pbar:
            if t % 1000 == 0:
                print(f'Episode {episode}, Timestep - {t}')
                logger.info(f'Episode {episode}, Timestep - {t}')
            learning_transitions = self.env.updateWhole(self.agent)
            estimated_delay = sum(self.env.dynetwork._delivery_times[-200:]) / (len(self.env.dynetwork._delivery_times[-200:])+1e-7)
            if t % self.training_freq == 0:
                loss_actor, loss_critic, rewards = self.agent.learn(learning_transitions)
            else:
                loss_actor = 0
                loss_critic = 0
                if len(learning_transitions) != 0:
                    rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                else:
                    rewards = 0
            pbar.set_postfix({'Reward': rewards, 'Delay': estimated_delay, 'Loss-Actor': loss_actor, 'Loss-Critic': loss_critic})
            '''store attributes for performance measures'''
            step.append(t)
            deliveries.append(copy.deepcopy(self.env.dynetwork._deliveries))

            if rewards is None:
                rewards = -np.inf

            ''' Save time-step trends to stat collector '''
            self.stat_collector.add_timestep_trends(self.env, self.name, loss_actor+loss_critic, rewards, estimationVsReturnError=[0])

            ''' Save time-step trends to tensorboard '''
            self.saveStatsToTensorBoard(rewards, episode, timestep=t, td_errors=loss_actor+loss_critic, delay=estimated_delay)

        print('Number of packets -> ', self.env.dynetwork._deliveries)
        '''Save all performance measures'''
        self.stat_collector.add_stats(self.env, self.name)
        print(self.env.calc_avg_delivery())
        self.env.reset_env()

class RLFedratedA2CTrainer(RLTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq):
        super().__init__(time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq)
        self.update_federated_freq = TARGET_UPDATE

    def train(self, episode):
        print("---------- Episode:", episode + 1, " ----------")
        step = []
        deliveries = []
        pbar = tqdm(range(self.time_steps))
        '''iterate each time step try to finish routing within time_steps'''
        if episode != 0:
            self.env.dynetwork._lambda_load = self.arrival_rate_load[episode % self.arrival_rate_load.shape[0]]
        for t in pbar:
            if t % 1000 == 0:
                print(f'Episode {episode}, Timestep - {t}')
                logger.info(f'Episode {episode}, Timestep - {t}')
            learning_transitions = self.env.updateWhole(self.agent)
            estimated_delay = sum(self.env.dynetwork._delivery_times[-200:]) / (len(self.env.dynetwork._delivery_times[-200:])+1e-7)
            if t % self.training_freq == 0:
                loss_actor, loss_critic, rewards = self.agent.learn(learning_transitions)
            else:
                loss_actor = 0
                loss_critic = 0
                if len(learning_transitions) != 0:
                    rewards = torch.cat(tuple(torch.tensor([exps[2]]) for exps in learning_transitions)).sum()
                else:
                    rewards = 0
            pbar.set_postfix({'Reward': rewards, 'Delay': estimated_delay, 'Loss-Actor': loss_actor, 'Loss-Critic': loss_critic})
            '''store attributes for performance measures'''
            step.append(t)
            deliveries.append(copy.deepcopy(self.env.dynetwork._deliveries))

            if rewards is None:
                rewards = -np.inf

            if t % self.update_federated_freq == 0:
                self.agent.update_central_controller_weights()

            ''' Save time-step trends to stat collector '''
            self.stat_collector.add_timestep_trends(self.env, self.name, loss_actor+loss_critic, rewards, estimationVsReturnError=[0])

            ''' Save time-step trends to tensorboard '''
            self.saveStatsToTensorBoard(rewards, episode, timestep=t, td_errors=loss_actor+loss_critic, delay=estimated_delay)

        print('Number of packets -> ', self.env.dynetwork._deliveries)
        '''Save all performance measures'''
        self.stat_collector.add_stats(self.env, self.name)
        print(self.env.calc_avg_delivery())
        self.env.reset_env()

class RLFuncApproximationTrainer(RLTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq):
        super().__init__(time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiment, update_freq)

    # def update_lr(self, epoch, schedule, gammas):
    #     for temp in range(0, len(schedule)):
    #         if schedule[temp] == epoch:
    #             for param_group in self.optimizer.param_groups:
    #                 self.current_lr = param_group['lr']
    #                 param_group['lr'] = self.current_lr * gammas[temp]
    #                 logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
    #                              self.current_lr * gammas[temp])
    #                 self.current_lr *= gammas[temp]

    def train(self, episode):
        print("---------- Episode:", episode + 1, " ----------")
        step = []
        deliveries = []
        pbar = tqdm(range(self.time_steps))
        mean_td_error = 0
        if episode != 0:
            self.env.dynetwork._lambda_load = self.arrival_rate_load[episode % self.arrival_rate_load.shape[0]]
        '''iterate each time step try to finish routing within time_steps'''
        for t in pbar:
            if t % 1000 == 0:
                print(f'Episode {episode}, Timestep - {t}')
                logger.info(f'Episode {episode}, Timestep - {t}')
            '''key function that obtain action and update Q-table'''
            rewards = self.env.updateWhole(self.agent)
            estimated_delay = sum(self.env.dynetwork._delivery_times[-200:]) / (len(self.env.dynetwork._delivery_times[-200:])+1e-7)
            pbar.set_postfix({'Reward': rewards, 'Delay': estimated_delay, 'Loss': mean_td_error, 'Epsilon': self.agent.config["epsilon"]})

            temporal_td_errors = []
            td_error_temp = self.agent.learn()
            # for destination_node in range(self.env.nnodes):
            #     ''' We pass None as our current node to avoid pushing into memory '''
            #
            if td_error_temp is not None:
                temporal_td_errors.append(td_error_temp)
            mean_td_error = np.mean(temporal_td_errors)
            self.agent.update_epsilon()
            '''store attributes for performance measures'''
            step.append(t)
            deliveries.append(copy.deepcopy(self.env.dynetwork._deliveries))

            if (t + 1) % self.target_update == 0:
                self.agent.update_target_weights()
            if rewards is None:
                rewards = -np.inf

            ''' Save time-step trends to stat collector '''
            self.stat_collector.add_timestep_trends(self.env, self.name, mean_td_error, rewards, estimationVsReturnError=[0])

            ''' Save time-step trends to tensorboard '''
            self.saveStatsToTensorBoard(rewards, episode, timestep=t, td_errors=mean_td_error, delay=estimated_delay)

        print('Number of packets -> ', self.env.dynetwork._deliveries)
        '''Save all performance measures'''
        self.stat_collector.add_stats(self.env, self.name)
        print(self.env.calc_avg_delivery())
        self.env.reset_env()

class RLGraphFuncApproximationTrainer(RLTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiemet):
        super().__init__(time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiemet)

    # def update_lr(self, epoch, schedule, gammas):
    #     for temp in range(0, len(schedule)):
    #         if schedule[temp] == epoch:
    #             for param_group in self.optimizer.param_groups:
    #                 self.current_lr = param_group['lr']
    #                 param_group['lr'] = self.current_lr * gammas[temp]
    #                 logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
    #                              self.current_lr * gammas[temp])
    #                 self.current_lr *= gammas[temp]

    def train(self, episode):
        print("---------- Episode:", episode + 1, " ----------")
        step = []
        deliveries = []
        pbar = tqdm(range(self.time_steps))
        loss_temp = 0
        '''iterate each time step try to finish routing within time_steps'''
        for t in pbar:
            if t % 1000 == 0:
                logger.info(f'Timestep - {t}')
            '''key function that obtain action and update Q-table'''
            rewards = self.env.updateWhole(self.agent)
            pbar.set_postfix({'Reward': rewards, 'Epsilon': self.agent.config["epsilon"], 'Loss': loss_temp})

            ''' We pass None as our current node to avoid pushing into memory '''
            loss_temp = self.agent.learn()

            ''' Update the exploration rate for all agents '''
            self.agent.update_epsilon()

            '''store attributes for performance measures'''
            step.append(t)
            deliveries.append(copy.deepcopy(self.env.dynetwork._deliveries))

            if (t + 1) % self.target_update == 0:
                self.agent.update_target_weights()

            self.saveStatsToTensorBoard(rewards, episode, timestep=t, td_errors=loss_temp)

        print('Number of packets -> ', self.env.dynetwork._deliveries)
        '''Save all performance measures'''
        self.stat_collector.add_stats(self.env, self.name)
        print(self.env.calc_avg_delivery())
        self.env.reset_env()

class RLGraphTopologyTrainer(RLTrainer):
    def __init__(self, time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiemet):
        super().__init__(time_steps, TARGET_UPDATE, agent, stat_collector, env, name, writer, experiemet)
        self.rl_agent = self.agent[0]
        self.optimal_agent = self.agent[1]

    # def update_lr(self, epoch, schedule, gammas):
    #     for temp in range(0, len(schedule)):
    #         if schedule[temp] == epoch:
    #             for param_group in self.optimizer.param_groups:
    #                 self.current_lr = param_group['lr']
    #                 param_group['lr'] = self.current_lr * gammas[temp]
    #                 logger.debug("Changing learning rate from %0.9f to %0.9f", self.current_lr,
    #                              self.current_lr * gammas[temp])
    #                 self.current_lr *= gammas[temp]
    def saveStatsToTensorBoard(self, rewards, episode, timestep, td_errors):
        self.writer.add_scalar('Reward', rewards, episode * self.time_steps + timestep)
        self.writer.add_scalar('SRC_Loss', td_errors['src'], episode * self.time_steps + timestep)
        self.writer.add_scalar('Dest_Loss', td_errors['dest'], episode * self.time_steps + timestep)

    def train(self, episode):
        print("---------- Episode:", episode + 1, " ----------")
        pbarK = tqdm(range(10))
        loss_temp = {'src':0, 'dest':0}
        '''iterate each time step try to finish routing within time_steps'''
        if self.optimal_agent is not None:
            optimal_score, random_score = self.optimal_agent.find_optimal_topology_score(self.env)
        else:
            optimal_score, random_score = None, None
        actions = []
        for k in pbarK:
            pbar = tqdm(range(self.time_steps))
            self.env.reset()
            list_action = []
            for t in pbar:
                ''' Get the current state '''
                state = self.env.state()

                ''' Apply action to the current state '''
                action, estimation = self.rl_agent.epsilon_greedy_policy(state)
                if action is None:
                    break
                list_action.append(action)
                ''' Step according the agent's decision '''
                next_state, reward, done = self.env.step(*action)

                pbar.set_postfix({'Reward': reward, 'Epsilon': self.rl_agent.epsilon, 'Src-Loss': loss_temp['src'], 'Dest-Loss': loss_temp['dest'], 'Q-Src-Estimation': estimation[0], 'Q-Dest-Estimation': estimation[1]})

                self.rl_agent.add_memory(state, action, next_state, torch.tensor(reward), torch.tensor(done))

                ''' We pass None as our current node to avoid pushing into memory '''
                loss_temp = self.rl_agent.learn()
                ''' Update the exploration rate for all agents '''
                self.rl_agent.update_epsilon()

                self.saveStatsToTensorBoard(reward, episode, timestep=t, td_errors=loss_temp)

                if done:
                    break
            self.writer.add_scalar('Graph Score', np.sum(self.env.calculateGraphDelayScore()), len(pbarK) * episode + k)
            if optimal_score is not None:
                self.writer.add_scalar('Graph Score Optimal Ratio',  optimal_score / np.sum(self.env.calculateGraphDelayScore()), len(pbarK) * episode + k)
                self.writer.add_scalar('Graph Score Random Ratio',   np.sum(self.env.calculateGraphDelayScore()) / random_score, len(pbarK) * episode + k)
            actions.append(list_action)
        print(actions)
        if (episode + 1) % self.target_update == 0:
            self.rl_agent.update_target_weights()
        # print('Number of packets -> ', self.env.dynetwork._deliveries)
        # '''Save all performance measures'''
        # self.stat_collector.add_stats(self.env, self.name)
        # print(self.env.calc_avg_delivery())
        self.env.reset_topology()