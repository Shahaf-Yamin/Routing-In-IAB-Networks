from Utils import Statistics
from Utils.ml_flow_utils import preprocess_meta_data
from Environment import env
from Agents import Agent
from trainer import trainer, evaluator
#import wandb
import logging
import os

logger = logging.getLogger("logger")
'''
Running Platform
'''
device = None
'''
Global LUTs
'''
envsLut = {'Q-Routing': env.dynetworkEnvQlearning, 'Full-Echo-Q-Routing': env.dynetworkEnvFullEchoQLearning,
           'Shortest-Path': env.dynetworkEnvShortestPath, 'Back-Pressure': env.dynetworkEnvBackPressure,
           'Random': env.dynetworkEnvRandom, 'Tabular-Actor-Critic': env.dynetworkEnvQlearning, 'Deep-Actor-Critic': env.dynetworkEnvA2cCentralized,
           'Relational-Actor-Critic': env.dynetworkEnvCentralizedA2cRelationalState, 'Decentralized-Relational-Actor-Critic': env.dynetworkEnvDeCentralizedA2cRelationalState,
           'Federated-Relational-Actor-Critic': env.dynetworkEnvFedCentralizedA2cRelationalState}

agentsLut = {'Q-Routing': Agent.QAgent, 'Full-Echo-Q-Routing': Agent.FullEchoQAgent, 'Shortest-Path': None, 'Back-Pressure': Agent.Backpressure_agent,
              'Random': Agent.RandomAgent, 'Tabular-Actor-Critic': Agent.HybridQ, 'Deep-Actor-Critic': Agent.CentralizedA2CAgent,
              'Relational-Actor-Critic': Agent.RelationalA2CAgent,'Decentralized-Relational-Actor-Critic': Agent.DecentralizedRelationalA2CAgent,
              'Federated-Relational-Actor-Critic': Agent.FedratedA2CAgent}

trainerLut = {'Q-Routing': trainer.RLTabularTrainer, 'Full-Echo-Q-Routing': trainer.RLTabularTrainer, 'Shortest-Path': None, 'Back-Pressure': None,
              'Random': None, 'Tabular-Actor-Critic': trainer.RLTabularTrainer, 'Deep-Actor-Critic': trainer.RLA2CTrainer, 'Relational-Actor-Critic': trainer.RLA2CTrainer,
              'Decentralized-Relational-Actor-Critic': trainer.RLA2CTrainer, 'Federated-Relational-Actor-Critic': trainer.RLFedratedA2CTrainer}

evaluatorLut = {'Q-Routing': evaluator.TabularRLRouterEvaluator, 'Full-Echo-Q-Routing': evaluator.TabularRLRouterEvaluator,
                'Shortest-Path': evaluator.RouterEvaluator, 'Back-Pressure': evaluator.RouterEvaluator,
                'Random': evaluator.RouterEvaluator, 'Tabular-Actor-Critic': evaluator.TabularRLRouterEvaluator,
                'Deep-Actor-Critic': evaluator.A2cRouterEvaluator, 'Relational-Actor-Critic': evaluator.A2cRouterEvaluator,
                'Decentralized-Relational-Actor-Critic': evaluator.A2cRouterEvaluator, 'Federated-Relational-Actor-Critic': evaluator.RLFedratedA2CEvaluator}

class Experiment:
    '''
    General class to store results of any experiment that trains an agent and test it afterwards over constant loads
    '''
    def __init__(self, setting, experiment):

        '''
        Result Statistics Initialization
        '''
        try:
            self.Env = envsLut[setting["NETWORK"]["Environment"]]
        except KeyError:
            raise Exception(f'Invalid Environment {setting["NETWORK"]["Environment"]} please use one of the following - {envsLut.keys()}')

        self.path = setting.result_dir
        self.train_path = self.path + '/train/'
        self.test_path = self.path + '/test/'
        self.results = {}
        self.name = setting["NETWORK"]["Environment"]
        self.experiment = experiment

        '''
        Simulation Initialization
        '''
        self.reward_func = setting["AGENT"]["rewardfunction"]
        self.TARGET_UPDATE = setting["Simulation"]["num_time_step_to_update_target_network"]
        self.env = self.Env(setting=setting, seed=setting["seed"], algorithm=self.name, rewardfun=self.reward_func)

        '''
        Agent Initialization
        '''
        try:
            self.state_space = self.env.get_state_space_dim(setting)
        except:
            self.state_space = None

        if agentsLut[setting["NETWORK"]["Environment"]] is not None:
            self.agent = agentsLut[setting["NETWORK"]["Environment"]](self.env.dynetwork, setting, self.state_space, device)
        else:
            self.agent = None

        '''
        Setting Configuration for both train and test modes
        '''
        self.stats_setting = {
                            'test':
                                  {"max_allowed_time_step_per_episode": setting["Simulation"]["test_allowed_time_step_per_episode"],
                                   "training_episodes": setting["Simulation"]["training_episodes"],
                                   "capacity": setting["NETWORK"]["holding capacity"], "test_trials_per_load": setting["Simulation"]["test_trials_per_load"],
                                   "num_nodes": setting["NETWORK"]["number Basestation"] + setting["NETWORK"]["number user"],
                                   "num_bs": setting["NETWORK"]["number Basestation"]},
                            'train':
                                  {"max_allowed_time_step_per_episode": setting["Simulation"]["max_allowed_time_step_per_episode"],
                                   "training_episodes": setting["Simulation"]["training_episodes"],
                                   "capacity": setting["NETWORK"]["holding capacity"],
                                   "num_nodes": setting["NETWORK"]["number Basestation"] + setting["NETWORK"]["number user"],
                                   "num_bs": setting["NETWORK"]["number Basestation"]}
                              }

    def store_results(self, keys=None):
        # Store results
        if keys is None:
            for key in self.results:
                logger.info(f'Dumping and Plotting {key} results')
                self.results[key].plot_result(loud=False)
                self.results[key].dump_statistics()
                del self.results[key]
        else:
            for key in keys:
                self.results[key].plot_result(loud=False)
                self.results[key].dump_statistics()
                del self.results[key]

class E2E_experiment(Experiment):
    '''
    Class to store results of any experiment that trains an agent and test it afterwards over constant loads
    '''

    def __init__(self, setting, experiment):

        '''
        Result Statistics Initialization
        '''
        super().__init__(setting, experiment)


        self.results['train'] = Statistics.TrainQLStatisticsCollector(setting=self.stats_setting['train'], result_dir=self.train_path, algorithms=[self.name])
        self.results['test'] = Statistics.TestStatisticsCollector(setting=self.stats_setting['test'], result_dir=self.test_path, algorithms=[self.name])

        if setting["AGENT"]["pretrained_path"] is not False and setting["NETWORK"]["Environment"] != "Shortest-Path" and setting["NETWORK"]["Environment"] != "Back-Pressure":
            self.agent.load_agent(setting["AGENT"]["pretrained_path"])

        '''
        Set up trainer
        '''
        trainer = trainerLut[setting["NETWORK"]["Environment"]]
        if trainer is not None:
            self.trainer = trainer(time_steps=setting["Simulation"]["max_allowed_time_step_per_episode"],
                                   TARGET_UPDATE=self.TARGET_UPDATE, agent=self.agent,
                                   stat_collector=self.results["train"], env=self.env, name=self.name, writer=setting.train_writer, experiment=self.experiment, update_freq=setting["AGENT"]["learning_freq"])
        else:
            self.trainer = None

        self.tester = evaluatorLut[setting["NETWORK"]["Environment"]](stat_collector=self.results["test"], env=self.env, setting=setting, writer=setting.test_writer, update_freq=setting["AGENT"]["learning_freq"])
        self.setting = setting

    def run_exp(self):
        cktpt_freq = self.setting["AGENT"]["checkpoint_frequency"]
        if self.trainer is not None and self.setting["AGENT"]["enable_train"] is True:
            for episode in range(self.setting["Simulation"]["training_episodes"]):
                self.trainer.train(episode)
                logger.info(f'Starting Episode {episode}')
                if episode % cktpt_freq == 0:
                    # Checkpoint to save the trained agent
                    self.agent.save_agent(self.path+'/train/')

            # Save the trained agent
            self.agent.save_agent(self.path+'/train/')
        logger.info(f'Starting Evaluation Against Different Loads')
        self.store_results(['train'])
        self.tester.evaluate(self.agent)
        logger.info(f'Dumping Results to Result Folder')
        self.store_results(['test'])

class OnlineChangingLoad_experiment(Experiment):
    '''
    Class to store results of any experiment that uses an agent against online changing load
    '''

    def __init__(self, setting, experiment):
        '''
        Result Statistics Initialization
        '''
        super().__init__(setting, experiment)
        self.results['test'] = Statistics.OnlineTestStatisticsCollector(setting=self.stats_setting["test"], result_dir=self.test_path, algorithms=[self.name])

        if setting["AGENT"]["pretrained_path"] is not False and setting["NETWORK"]["Environment"] != "Shortest-Path" and setting["NETWORK"]["Environment"] != "Back-Pressure":
            self.agent.load_agent(setting["AGENT"]["pretrained_path"])
        elif setting["NETWORK"]["Environment"] != "Shortest-Path" and setting["NETWORK"]["Environment"] != "Back-Pressure" and setting["NETWORK"]["Environment"] != 'Random':
            raise Exception('Online Experiment is supported only with a trained agent!')
        else:
            pass

        self.tester = evaluatorLut[setting["NETWORK"]["Environment"]](stat_collector=self.results["test"], env=self.env, setting=setting, writer=setting.test_writer, update_freq=setting["AGENT"]["learning_freq"])
        self.setting = setting

    def run_exp(self):
        self.tester.evaluateChangingLoad(self.agent)
        self.store_results()

class NodeFailure_experiment(Experiment):
    '''
    Class to store results of any experiment that uses an agent against online changing load
    '''

    def __init__(self, setting, experiment):
        '''
        Result Statistics Initialization
        '''
        super().__init__(setting, experiment)
        self.results['test'] = Statistics.OnlineTestStatisticsCollector(setting=self.stats_setting["test"], result_dir=self.test_path, algorithms=[self.name])

        if setting["AGENT"]["pretrained_path"] is not False and setting["NETWORK"]["Environment"] != "Shortest-Path" and\
                setting["NETWORK"]["Environment"] != "Back-Pressure" and setting["NETWORK"]["Environment"] != 'Random':
            self.agent.load_agent(setting["AGENT"]["pretrained_path"])
        elif setting["NETWORK"]["Environment"] != "Shortest-Path" and setting["NETWORK"]["Environment"] != "Back-Pressure" and setting["NETWORK"]["Environment"] != 'Random':
            raise Exception('Online Experiment is supported only with a trained agent!')
        else:
            pass

        self.tester = evaluatorLut[setting["NETWORK"]["Environment"]](stat_collector=self.results["test"], env=self.env, setting=setting, writer=setting.test_writer, update_freq=setting["AGENT"]["learning_freq"])
        self.setting = setting

    def run_exp(self):
        self.tester.evaluateNodeFailure(self.agent)
        self.store_results()

class Topology_experiment(Experiment):
    '''
    Class to store results of different topologies performance under the same number of parents and childrens
    '''

    def __init__(self, setting, experiment):
        '''
        Result Statistics Initialization
        '''
        super().__init__(setting, experiment)

        self.results['train'] = Statistics.TrainQLStatisticsCollector(setting=self.stats_setting['train'], result_dir=self.train_path, algorithms=[self.name])
        self.results['test'] = Statistics.TestStatisticsCollector(setting=self.stats_setting['test'], result_dir=self.test_path, algorithms=[self.name])

        if setting["AGENT"]["pretrained_path"] is not False and setting["NETWORK"]["Environment"] != "Shortest-Path" and setting["NETWORK"]["Environment"] != "Back-Pressure":
            self.agent.load_agent(setting["AGENT"]["pretrained_path"])

        self.tester = evaluator.TopologyRouterEvaluator(stat_collector=self.results["test"], env=self.env, setting=setting)
        self.setting = setting

    def run_exp(self):
        self.tester.evaluate(self.agent)
        self.store_results()

if __name__ == "__main__":
    # Parse manual arguments
    os.chdir('Experiments')
    setting, args, temp_device, experiment = preprocess_meta_data()
    device = temp_device

    if args.mode == 0:
        exp = E2E_experiment(setting=setting, experiment=experiment)
    elif args.mode == 1:
        exp = OnlineChangingLoad_experiment(setting=setting, experiment=experiment)
    elif args.mode == 2:
        exp = Topology_experiment(setting=setting, experiment=experiment)
    elif args.mode == 3:
        exp = NodeFailure_experiment(setting=setting, experiment=experiment)
    else:
        raise Exception('Invalid Running Mode')

    exp.run_exp()
