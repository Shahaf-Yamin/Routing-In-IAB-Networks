import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import pickle
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import matplotlib as mpl
import seaborn as sns
import os
from scipy.stats import entropy

mpl.rcParams['agg.path.chunksize'] = 10000

def get_exp_names():
    return ["Average Delay Time", "Maximum Number of Packets a Single Node Hold", "Average Number of Packets a Single Node Hold",
                          "Percentile of Working Nodes at Maximal Capacity", "Percentile of Non-Empty Nodes",
                          "Average Normalized Throughput", "Average Link Utilization", "Average Transmission Rejections", "Arrival Ratio", "Fairness", "Packet Drop Trend",
                            "Arrival Rate Trend",'Circles Count', 'Hop Count','Load', "Packet Drop Percentile"]

def get_ylabel_vec():
    return  ['Average Delay Time [Time-Steps]', 'Packets', 'Packets', 'Percentile of Working Nodes at Maximal Capacity',
                           'Percentile of Non-Empty Nodes', 'Average Normalized Throughput', 'Average Link Utilization',
                           "Average Transmission Rejections", "Arrival Ratio", "Fairness", '#Packets', 'Arrival Rate Trend',
                            'Circles Count', 'Hop Count', 'Load', 'Packet Drop Percentile']

class StatisticsCollector(object):
    def __init__(self, setting, result_dir, algorithms):
        self.exp_names = get_exp_names()
        self.ylabel_vec = get_ylabel_vec()

        self.maximal_timestep = setting["max_allowed_time_step_per_episode"]
        self.maximalNumEpisode = setting["training_episodes"]
        self.capacity = setting["capacity"]
        self.result_dir = result_dir
        self.algorithms = algorithms

        self.stats = {exp: {} for exp in self.exp_names}
        for algorithm in algorithms:
            for exp in self.stats:
                self.stats[exp][algorithm] = np.zeros((self.maximalNumEpisode))
        self.curEpisode = {exp: 0 for exp in algorithms}
        self.curTimeSlot = {exp: 0 for exp in algorithms}
        self.EpisodeVec = np.arange(0, self.maximalNumEpisode)
        self.TimestepVec = np.arange(0, self.maximal_timestep)

    def add_stats(self, env, algorithm):
        curEpisode = self.curEpisode[algorithm]
        self.stats["Average Delay Time"][algorithm][curEpisode] = env.calc_avg_delivery()
        self.stats["Maximum Number of Packets a Single Node Hold"][algorithm][curEpisode] = env.dynetwork._max_queue_length
        self.stats["Average Number of Packets a Single Node Hold"][algorithm][curEpisode] = np.mean(env.dynetwork._avg_q_len_arr)
        self.stats["Percentile of Working Nodes at Maximal Capacity"][algorithm][curEpisode] = (np.array(env.dynetwork._num_capacity_node) / np.array(env.dynetwork._num_working_node)).mean() * 100
        self.stats["Percentile of Non-Empty Nodes"][algorithm][curEpisode] = (((np.sum(env.dynetwork._num_working_node)) / env.nnodes) / self.maximal_timestep) * 100
        self.stats["Average Normalized Throughput"][algorithm][curEpisode] = np.mean(env.dynetwork._avg_throughput)
        self.stats["Average Link Utilization"][algorithm][curEpisode] = np.mean(env.dynetwork._avg_link_utilization)
        self.stats["Average Transmission Rejections"][algorithm][curEpisode] = env.dynetwork._rejections / (env.dynetwork._initializations + env.npackets)
        self.stats["Arrival Ratio"][algorithm][curEpisode] = env.dynetwork._deliveries / (env.dynetwork._initializations + env.npackets)
        self.stats["Fairness"][algorithm][curEpisode] = np.nanmean(env.dynetwork._avg_router_fairness)
        self.stats["Packet Drop Percentile"][algorithm][curEpisode] = np.sum(env.dropped_packet_histogram) / (env.dynetwork._initializations + env.npackets)
        self.curEpisode[algorithm] += 1

    def plot_result(self, loud=False, colors=None):
        for exp, ylabel in zip(self.exp_names, self.ylabel_vec):
            if loud:
                print(f"{exp}")
            try:
                plt.clf()
                plt.title(f"{exp} vs Episode")
                for algorithm in self.algorithms:
                    if loud:
                        print(f"{algorithm}: {self.stats[exp][algorithm]}")
                    if colors is not None:
                        plt.plot(self.EpisodeVec, self.stats[exp][algorithm], c=colors[algorithm], label=algorithm)
                    else:
                        plt.plot(self.EpisodeVec, self.stats[exp][algorithm], label=algorithm)
                plt.xlabel('Episode')
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid()
                plt.savefig(self.result_dir + f"plots/{exp}.png", dpi=1024)
            except ValueError:
                plt.clf()

    def dump_statistics(self):
        for algorithm in self.algorithms:
            results_df = []
            for exp in self.exp_names:
                results_df.append(pd.DataFrame(list(self.stats[exp][algorithm]), columns=[f'Results']))
            final_res = pd.concat(results_df, axis=0, keys=self.exp_names)
            final_res.to_pickle(self.result_dir + f"data/Train_Statistic_{algorithm}.pkl")
            final_res.to_csv(self.result_dir + f"data/Train_Statistic_{algorithm}.csv")

class TrainQLStatisticsCollector(StatisticsCollector):
    def __init__(self, setting, result_dir, algorithms):
        super(TrainQLStatisticsCollector, self).__init__(setting, result_dir, algorithms)
        self.stats['TD_error'] = {}
        self.stats['Reward'] = {}
        self.stats['Return_Vs_Estimation'] = {}
        self.stats['Average Delay Trend'] = {}
        self.stats['Arrival Rate Trend'] = {}
        self.stats['Packet Drop Trend'] = {}
        self.stats['Circles Count'] = {}
        self.stats['Hop Count'] = {}
        self.stats['InformationExchange'] = {}
        self.stats['Return_Vs_Estimation'] = {}
        for algorithm in algorithms:
            self.stats['TD_error'][algorithm] = []
            self.stats['Reward'][algorithm] = []
            self.stats['Average Delay Trend'][algorithm] = []
            self.stats['Arrival Rate Trend'][algorithm] = []
            self.stats['Circles Count'][algorithm] = []
            self.stats['Hop Count'][algorithm] = []
            self.stats['Packet Drop Trend'][algorithm] = []
            self.stats['Load'][algorithm] = []
            self.stats['InformationExchange'][algorithm] = []
            self.stats['Return_Vs_Estimation'][algorithm] = []

    def add_timestep_trends(self, env, algorithm, td_errors, reward, estimationVsReturnError, informationExchange=0):
        try:
            self.stats['Load'][algorithm].append(env.dynetwork._lambda_load)
            self.stats['TD_error'][algorithm].append(np.mean(td_errors))
            self.stats['Reward'][algorithm].append(reward)
            self.stats['Return_Vs_Estimation'][algorithm].append(np.mean(estimationVsReturnError))
            self.stats['Average Delay Trend'][algorithm].append(sum(env.dynetwork._delivery_times[-200:]) / len(env.dynetwork._delivery_times[-200:]))
            self.stats['Packet Drop Trend'][algorithm].append(sum(env.dynetwork._packet_drops[-200:]) / len(env.dynetwork._packet_drops[-200:]))
            self.stats['Arrival Rate Trend'][algorithm].append(sum(env.dynetwork._packet_arrival[-200:]) / len(env.dynetwork._packet_arrival[-200:]))
            self.stats['Circles Count'][algorithm].append(sum(env.dynetwork._circles_counter[-200:]) / len(env.dynetwork._circles_counter[-200:]))
            self.stats['Hop Count'][algorithm].append(sum(env.dynetwork._hops_counter[-200:]) / len(env.dynetwork._hops_counter[-200:]))
            self.stats['InformationExchange'][algorithm].append(informationExchange)

        except ZeroDivisionError:
            self.stats["Average Delay Trend"][algorithm].append(0)
            self.stats["Packet Drop Trend"][algorithm].append(0)
            self.stats["Arrival Rate Trend"][algorithm].append(0)

        self.curTimeSlot[algorithm] += 1

    def plot_result(self, loud=False, colors=None):
        def plot_train_stats(stats, xlabels, ylabels, titles, withWindows):
            for stat, xlabel, ylabel, title, window in zip(stats, xlabels, ylabels, titles, withWindows):
                try:
                    for algorithm in self.algorithms:
                        if window:
                            signal = np.convolve(WINDOW, stat[algorithm], mode='same')
                        else:
                            signal = stat[algorithm]
                        if loud:
                            print(f"{algorithm}: {self.stats['TD_error'][algorithm]}")
                        if colors is not None:
                            plt.plot(list(range(0, len(stat[algorithm]))), signal, label=f'{algorithm}',c=colors[algorithm])
                        else:
                            plt.plot(list(range(0, len(stat[algorithm]))), signal, label=f'{algorithm}')
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.legend()
                    plt.grid()
                    plt.savefig(self.result_dir + f"plots/{title}.png", dpi=1024)
                    plt.clf()
                except ValueError:
                    pass
        super(TrainQLStatisticsCollector, self).plot_result(loud, colors)
        plt.clf()
        WINDOW_LENGTH = 20
        WINDOW = np.ones(WINDOW_LENGTH) / WINDOW_LENGTH
        xlabels = ['Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep']
        ylabels = [r'$\mathbb{E}[\delta_{TD}]$', r'${E}_{\pi}[\sum_{n=0}^{N-1}R^{(n)}_t]$', 'Delay', '#Packets', '#Packets', r'${E}_{\pi}[G_t-Q^{\pi}(S_t,A_t)]$', 'Bits']
        stats = [self.stats['TD_error'], self.stats['Reward'], self.stats['Average Delay Trend'], self.stats["Packet Drop Trend"], self.stats["Arrival Rate Trend"], self.stats["Return_Vs_Estimation"], self.stats["InformationExchange"]]
        titles = ['TD Error Per Time Slot', 'Rewards Per Time Slot', 'Average Delay Per Time Slot', 'Packet Drop Per Time Slot', 'Arrival Rate Per Time Slot', 'Return Vs Estimation Per Time Slot', 'Information Exchange Per Time Slot']
        withWindow = [True, True, False, False, False, True, True]
        plot_train_stats(stats, xlabels=xlabels, ylabels=ylabels, titles=titles, withWindows=withWindow)

    def dump_statistics(self):
        for algorithm in self.algorithms:
            results_df = []
            for exp in self.stats.keys():
                results_df.append(pd.DataFrame(list(self.stats[exp][algorithm]), columns=[f'Results']))
            final_res = pd.concat(results_df, axis=0, keys=self.stats.keys())
            final_res.to_pickle(self.result_dir + f"data/Train_Statistic_{algorithm}.pkl")
            final_res.to_csv(self.result_dir + f"data/Train_Statistic_{algorithm}.csv")

class TestStatisticsCollector(StatisticsCollector):
    def __init__(self, setting, result_dir, algorithms):
        super(TestStatisticsCollector, self).__init__(setting, result_dir, algorithms)
        self.stats = {exp: {} for exp in self.exp_names}
        self.histogrm_stats = {"Relay Histogram": {}, "Packet Generation Histogram": {}, "Packet Arrival Histogram": {}, "Packet Dropped Histogram": {}, "Packet Destination Histogram": {}}
        self.heatmap_stats = {"Path History": {}, "Path Drops History": {}, "Policy History": {}}
        for algorithm in algorithms:
            for exp in self.stats:
                self.stats[exp][algorithm] = {}
            self.histogrm_stats["Relay Histogram"][algorithm] = {}
            self.histogrm_stats["Packet Generation Histogram"][algorithm] = {}
            self.histogrm_stats["Packet Arrival Histogram"][algorithm] = {}
            self.histogrm_stats["Packet Dropped Histogram"][algorithm] = {}
            self.histogrm_stats["Packet Destination Histogram"][algorithm] = {}
            self.heatmap_stats["Path History"][algorithm] = {}
            self.heatmap_stats["Path Drops History"][algorithm] = {}
            self.heatmap_stats["Policy History"][algorithm] = {}

        self.algorithms = algorithms
        self.trials = setting["test_trials_per_load"] * 2
        self.nnodes = setting["num_nodes"]
        self.numBs = setting["num_bs"]
        self.numUe = self.nnodes - self.numBs
        self.numDest = self.numUe if self.numUe != 0 else self.numBs
        self.load_vec = []

    def init_new_load(self, load):
        self.cur_load = load
        if self.cur_load not in self.load_vec:
            self.load_vec.append(load)

            for algorithm in self.algorithms:
                for exp in self.exp_names:
                    self.stats[exp][algorithm][load] = np.zeros(self.trials)
                self.histogrm_stats["Relay Histogram"][algorithm][load] = np.zeros((self.trials, self.numBs))
                self.histogrm_stats["Packet Generation Histogram"][algorithm][load] = np.zeros((self.trials, self.numBs))
                self.histogrm_stats["Packet Arrival Histogram"][algorithm][load] = np.zeros((self.trials, self.numDest))
                self.histogrm_stats["Packet Destination Histogram"][algorithm][load] = np.zeros((self.trials, self.numDest))
                self.histogrm_stats["Packet Dropped Histogram"][algorithm][load] = np.zeros((self.trials, self.numBs))
                self.heatmap_stats["Path History"][algorithm][load] = np.zeros((self.trials, self.numBs, self.numDest))
                self.heatmap_stats["Path Drops History"][algorithm][load] = np.zeros((self.trials, self.numBs, self.numDest))
                self.heatmap_stats["Policy History"][algorithm][load] = np.zeros((self.trials, self.numDest, self.numBs, self.numUe+self.numBs))

    def add_load_trial_results(self, env, trial, algorithm):
        self.stats["Average Delay Time"][algorithm][self.cur_load][trial] = np.mean(env.dynetwork._delivery_times)
        self.stats["Maximum Number of Packets a Single Node Hold"][algorithm][self.cur_load][trial] = env.dynetwork._max_queue_length
        self.stats["Average Number of Packets a Single Node Hold"][algorithm][self.cur_load][trial] = np.mean(env.dynetwork._avg_q_len_arr)
        self.stats["Percentile of Working Nodes at Maximal Capacity"][algorithm][self.cur_load][trial] = (np.array(env.dynetwork._num_capacity_node) / np.array(env.dynetwork._num_working_node)).mean() * 100
        self.stats["Percentile of Non-Empty Nodes"][algorithm][self.cur_load][trial] =  (((np.sum(env.dynetwork._num_working_node)) / env.nnodes) / self.maximal_timestep) * 100
        self.stats["Average Normalized Throughput"][algorithm][self.cur_load][trial] = np.mean(env.dynetwork._avg_throughput)
        self.stats["Average Link Utilization"][algorithm][self.cur_load][trial] = np.mean(env.dynetwork._avg_link_utilization)
        self.stats["Average Transmission Rejections"][algorithm][self.cur_load][trial] = env.dynetwork._rejections/(env.dynetwork._generations + env.npackets)
        self.stats["Arrival Ratio"][algorithm][self.cur_load][trial] = env.dynetwork._deliveries/(env.dynetwork._generations + env.npackets)
        self.stats["Fairness"][algorithm][self.cur_load][trial] = np.nanmean(env.dynetwork._avg_router_fairness)
        self.stats["Packet Drop Percentile"][algorithm][self.cur_load][trial] = np.sum(env.dropped_packet_histogram) / (env.dynetwork._generations + env.npackets)
        self.histogrm_stats["Relay Histogram"][algorithm][self.cur_load][trial] = env.relayed_packet_histogram
        self.histogrm_stats["Packet Arrival Histogram"][algorithm][self.cur_load][trial] = env.arrived_packet_histogram
        self.histogrm_stats["Packet Dropped Histogram"][algorithm][self.cur_load][trial] = env.dropped_packet_histogram
        self.histogrm_stats["Packet Generation Histogram"][algorithm][self.cur_load][trial] = env.dynetwork.generated_packet_histogram
        self.histogrm_stats["Packet Destination Histogram"][algorithm][self.cur_load][trial] = env.dynetwork.generated_destination_histogram
        self.heatmap_stats["Path History"][algorithm][self.cur_load][trial] = env.pathMapping
        self.heatmap_stats["Path Drops History"][algorithm][self.cur_load][trial] = env.dropMapping
        self.heatmap_stats["Policy History"][algorithm][self.cur_load][trial] = env.PolicyMapping
        self.stats['Circles Count'][algorithm][self.cur_load][trial] = np.mean(env.dynetwork._circles_counter)
        self.stats['Hop Count'][algorithm][self.cur_load][trial] = np.mean(env.dynetwork._hops_counter)

    def dump_statistics(self):
        for algorithm in self.algorithms:
            results_df = []
            for exp in self.exp_names:
                results_df.append(pd.DataFrame(list(self.stats[exp][algorithm].items()), columns=['Load', f'Results']))
            final_res = pd.concat(results_df, axis=0, keys=self.exp_names)
            final_res.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}.pkl")
            final_res.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}.csv")
            RX_df = pd.DataFrame(list(self.histogrm_stats["Relay Histogram"][algorithm].items()), columns=['Load', f'Results'])
            RX_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}_Relay_histogram.csv")
            RX_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}_Relay_histogram.pkl")
            TX_df = pd.DataFrame(list(self.histogrm_stats["Packet Generation Histogram"][algorithm].items()), columns=['Load', f'Results'])
            TX_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}_Packet_Generation_Histogram.csv")
            TX_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}_Packet_Generation_Histogram.pkl")
            Arrival_df = pd.DataFrame(list(self.histogrm_stats["Packet Arrival Histogram"][algorithm].items()), columns=['Load', f'Results'])
            Arrival_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}_Packet_Arrival_histogram.csv")
            Arrival_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}_Packet_Arrival_histogram.pkl")
            Drop_df = pd.DataFrame(list(self.histogrm_stats["Packet Dropped Histogram"][algorithm].items()), columns=['Load', f'Results'])
            Drop_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Dropped_Histogram.csv")
            Drop_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Dropped_Histogram.pkl")
            Destination_df = pd.DataFrame(list(self.histogrm_stats["Packet Destination Histogram"][algorithm].items()), columns=['Load', f'Results'])
            Destination_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Destination_Histogram.csv")
            Destination_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Destination_Histogram.pkl")
            Heatmap_df = pd.DataFrame(list(self.heatmap_stats["Path History"][algorithm].items()), columns=['Load', f'Results'])
            Heatmap_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Destination_Histogram.csv")
            Heatmap_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Destination_Histogram.pkl")
            Heatmap_df = pd.DataFrame(list(self.heatmap_stats["Path Drops History"][algorithm].items()), columns=['Load', f'Results'])
            Heatmap_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Destination_Histogram.csv")
            Heatmap_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}Packet_Destination_Histogram.pkl")
            Heatmap_df = pd.DataFrame(list(self.heatmap_stats["Policy History"][algorithm].items()), columns=['Load', f'Results'])
            Heatmap_df.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}Policy_Histogram.csv")
            Heatmap_df.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}Policy_Histogram.pkl")

    def plot_result(self, loud=False, colors=None):
        def plot_histogram(load_results, title, xlabel, x_axis):
            X, Y = np.meshgrid(x_axis, self.load_vec)
            Z = np.zeros_like(X)
            for load_idx, load in enumerate(self.load_vec):
                average_result = load_results["Results"][load_idx].mean(axis=0)
                for router in range(average_result.shape[0]):
                    Z[load_idx][router] = average_result[router]
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

            surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=True)
            ax.zaxis.set_major_locator(LinearLocator(5))
            ax.set_xlabel(xlabel)
            ax.set_ylabel('Load')
            ax.set_zlabel('Packets')
            ax.set_title(f'{title}')


        # Plot Heatmap Per Load

        # Plot trends
        for exp, ylabel in zip(self.exp_names, self.ylabel_vec):
            if loud:
                print(f"{exp}")
            plt.clf()
            plt.title(f"{exp} vs Network Load")
            for algorithm in self.algorithms:
                if loud:
                    print(f"{algorithm}: {self.stats[exp][algorithm]}")
                mean = np.array(list(map(lambda x: np.mean(x[1]), self.stats[exp][algorithm].items())))
                std = np.array(list(map(lambda x: np.std(x[1]), self.stats[exp][algorithm].items())))
                low_std = mean - std
                low_std[low_std < 0] = 0
                if colors is not None:
                    plt.plot(self.load_vec, mean, c=colors[algorithm], label=algorithm)
                    plt.fill_between(self.load_vec, low_std, mean + std, alpha=0.25, facecolor=colors[algorithm])
                else:
                    plt.plot(self.load_vec, mean, label=algorithm)
                    plt.fill_between(self.load_vec, low_std, mean + std, alpha=0.25)

            plt.xlabel('Network Load [Packets]')
            plt.ylabel(ylabel)
            plt.legend()
            plt.grid()
            plt.savefig(self.result_dir + f"plots/{exp}.png", dpi=1024)
            plt.clf()

        # Plot 3D Histogram
        for algorithm in self.algorithms:
            for title in ["Relay Histogram", "Packet Generation Histogram", "Packet Arrival Histogram", "Packet Dropped Histogram", "Packet Destination Histogram"]:
                df = pd.DataFrame(list(self.histogrm_stats[title][algorithm].items()), columns=['Load', f'Results'])
                plt.figure()
                x_label = 'Destination' if title == "Packet Arrival Histogram" else 'Relay'
                x_axis = np.arange(0, df['Results'][0].shape[1])
                plot_histogram(df, title, x_label, x_axis)
                plt.savefig(self.result_dir + f"plots/{algorithm}_{title}.png")
                plt.clf()
        # Plot Heatmaps and Box-plots
        sns.set_theme()
        stat_visual_path = self.result_dir + 'stat_visualizations'
        if not os.path.exists(stat_visual_path):
            os.makedirs(stat_visual_path)
        if not os.path.exists(stat_visual_path+'/Policy'):
            os.makedirs(stat_visual_path+'/Policy')
            for agent in range(self.numBs):
                os.makedirs(stat_visual_path + f'/Policy/bs_{agent}')
        for algorithm in self.algorithms:
            for load in self.load_vec:
                plt.figure()
                ax = sns.heatmap(np.mean(self.heatmap_stats["Path History"][algorithm][load], axis=0))
                ax.set_title(f'Packet Path Heatmap Load - {load}')
                ax.set_xlabel('Users')
                ax.set_ylabel('Base-Stations')
                plt.savefig(stat_visual_path + f"/{algorithm}_pathHeatmap_Load_{load}.png")
                plt.clf()
                plt.figure()
                ax = sns.heatmap(np.mean(self.heatmap_stats["Path Drops History"][algorithm][load], axis=0))
                ax.set_title(f'Packet Drop Heatmap Load - {load}')
                ax.set_xlabel('Users')
                ax.set_ylabel('Base-Stations')
                plt.savefig(stat_visual_path + f"/{algorithm}_pathDropsmap_Load_{load}.png")
                plt.clf()
                pathBsStatistic = pd.DataFrame(data=self.heatmap_stats["Path History"][algorithm][load].sum(axis=2),
                                               index=[f'MC {idx}' for idx in range(self.heatmap_stats["Path History"][algorithm][load].shape[0])],
                                               columns=[f'Bs {idx}' for idx in range(self.heatmap_stats["Path History"][algorithm][load].shape[1])])

                dropsBsStatistic = pd.DataFrame(data=self.heatmap_stats["Path Drops History"][algorithm][load].sum(axis=2),
                                                index=[f'MC {idx}' for idx in
                                                       range(self.heatmap_stats["Path Drops History"][algorithm][load].shape[0])],
                                                columns=[f'Bs {idx}' for idx in
                                                         range(self.heatmap_stats["Path Drops History"][algorithm][load].shape[1])])

                plt.figure()
                ax = sns.boxplot(data=pathBsStatistic)
                ax.set_ylabel('Path Involvement [Packets]')
                ax.set_title(f'Path Involvement Per BaseStation Load - {load}')
                plt.savefig(stat_visual_path + f"/Path_Involvement_Boxplot_Basestation_load_{load}.png")
                plt.figure()

                ax = sns.boxplot(data=dropsBsStatistic)
                ax.set_ylabel('Drops [Packets]')
                ax.set_title(f'Packet Drops Involvement Per BaseStation Load - {load}')
                plt.savefig(stat_visual_path + f"/Packet_Drops_Involvement_Boxplot_Basestation_load_{load}.png")

                for agent in range(self.numBs):
                    plt.figure()
                    ax = sns.heatmap(np.mean(self.heatmap_stats["Policy History"][algorithm][load], axis=0)[:, agent, :])
                    ax.set_title(f'Policy Heatmap Load - {load} - Basestation {agent}')
                    ax.set_xlabel('Actions')
                    ax.set_ylabel('States')
                    plt.savefig(stat_visual_path + f"/Policy/bs_{agent}/{algorithm}_policy_Load_{load}.png")
                    plt.clf()

            entropyDataFrame = pd.DataFrame()
            for load in self.load_vec:
                # Take average over the monte carlo axis and get policies with shape (state, agent, action)
                meanPolicies = np.mean(self.heatmap_stats["Policy History"][algorithm][load], axis=0)
                # Calculate policy probabilities in the form of (action, agent, state)
                normalizedAgentPolicy = meanPolicies.T / (np.sum(meanPolicies, axis=2).T+1e-9)
                # Calculate Shannon's Entropy based on the policies distribution
                AgentEntropy = entropy(normalizedAgentPolicy, base=2, axis=0)
                # Calcualte the State probability to be in a specific state
                agentStateHistogram = normalizedAgentPolicy.sum(axis=0)
                # Calculate the average entropy over the state dimension per agent without considering NaN Values
                agentStateProbability = (agentStateHistogram.T / (1e-9 + np.sum(agentStateHistogram, axis=1))).T
                AgentEntropyMaskedNaN = np.ma.MaskedArray(AgentEntropy, mask=np.isnan(AgentEntropy))
                try:
                    AverageEntropy = np.average(AgentEntropyMaskedNaN, axis=1, weights=agentStateProbability)
                except ZeroDivisionError:
                    continue
                # Append the results to entropy list
                entropyDataFrame[load] = AverageEntropy._data

            ax = sns.boxplot(data=entropyDataFrame)
            ax.set_ylabel('Entropy')
            ax.set_xlabel('Load')
            ax.set_title(f'Policies Entropy Per Load')
            plt.savefig(stat_visual_path + f"/Policy/entropy.png")

class TrainStatisticsLoader():
    def __init__(self, result_dir):
        self.exp_names = ['Average Delay Trend', 'Reward', 'TD_error', 'Packet Drop Trend', 'Arrival Rate Trend', 'Return_Vs_Estimation']
        # Initialize the stats dictionary
        self.stats = {exp: {} for exp in self.exp_names}
        self.load_vec = {}
        self.algorithms = []
        self.result_dir = result_dir

    def add_data(self, path, algorithm):
        currData = pd.read_pickle(path)
        self.algorithms.append(algorithm)
        # Set up the corresponding data structure
        for exp in self.exp_names:
            self.stats[exp][algorithm] = currData["Results"][exp]

    def append_data(self, path, algorithm, seed):

        if seed == 0:
            self.add_data(path, algorithm)
        else:
            currData = pd.read_pickle(path)
            for exp in self.exp_names:
                currLoadRes = currData["Results"][exp]
                self.stats[exp][algorithm] = np.vstack((self.stats[exp][algorithm], currLoadRes))

    def plot_multiple_runs_data(self, loud, colors=None):

        def plot_train_stats(stats, xlabels, ylabels, titles, withWindows):
            for stat, xlabel, ylabel, title, window in zip(stats, xlabels, ylabels, titles, withWindows):
                for algorithm in self.algorithms:
                    print(f'{title}:{algorithm}')
                    if window:
                        for idx in range(stat[algorithm].shape[0]):
                            print(idx)
                            try:
                                stat[algorithm][idx] = np.convolve(WINDOW, np.array([np.array(elem) if not np.isneginf(elem) else 0 for elem in stat[algorithm][idx]]), mode='same')
                            except TypeError:
                                print(1)
                    signal = np.nanmean(stat[algorithm].astype(np.float64), axis=0)
                    std = np.nanstd(stat[algorithm].astype(np.float64), axis=0)
                    low_std = signal - std
                    high_std = signal + std

                    if colors is not None:
                        if title == 'TD Error Per Time Slot':
                            plt.semilogy(list(range(0, signal.shape[0])), signal, c=colors[algorithm], label=algorithm)
                        else:
                            plt.plot(list(range(0, signal.shape[0])), signal, c=colors[algorithm], label=algorithm)
                        # plt.fill_between(list(range(0, signal.shape[0])), low_std, high_std, alpha=0.25,
                        #                  facecolor=colors[algorithm])
                    else:
                        if title == 'TD Error Per Time Slot':
                            plt.semilogy(list(range(0, signal.shape[0])), signal, label=algorithm)
                        else:
                            plt.plot(list(range(0, signal.shape[0])), signal, label=algorithm)
                        # plt.fill_between(list(range(0, signal.shape[0])), low_std, high_std, alpha=0.25)

                plt.title(title)
                plt.xlabel(xlabel)
                plt.ylabel(ylabel)
                plt.legend()
                plt.grid()
                plt.xlim([0, stat[self.algorithms[0]].shape[1]-WINDOW_LENGTH])
                plt.ylim([20, 150])

                plt.savefig(self.result_dir + f"plots/{title}.png", dpi=1024)
                plt.clf()

        plt.clf()
        WINDOW_LENGTH = 1000
        WINDOW = np.ones(WINDOW_LENGTH) / WINDOW_LENGTH
        xlabels = ['Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep']
        ylabels = ['Delay', r'${E}_{\pi}[\sum_{n=0}^{N-1}R^{(n)}_t]$',r'$\mathbb{E}[\delta_{TD}]$', '#Packets',
                   '#Packets', r'${E}_{\pi}[G_t-Q^{\pi}(S_t,A_t)]$']
        stats = [self.stats[exp] for exp in self.exp_names]
        titles = ['Average Delay Per Time Slot', 'Rewards Per Time Slot', 'TD Error Per Time Slot',
                  'Packet Drop Per Time Slot', 'Packets Arrival Per Time Slot', 'Return Vs Estimation Per Time Slot']
        withWindow = [True, True, True, True, True, True]
        plot_train_stats(stats, xlabels=xlabels, ylabels=ylabels, titles=titles, withWindows=withWindow)

    def plot_data(self, loud, colors=None):
        mStyles = ["*", "h", "o", "v", ".", ",", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        def plot_train_stats(stats, xlabels, ylabels, titles, withWindows):
            for stat, xlabel, ylabel, title, window in zip(stats, xlabels, ylabels, titles, withWindows):
                try:
                    for idx, algorithm in enumerate(self.algorithms):
                        if window:
                            signal = np.convolve(WINDOW, stat[algorithm], mode='same')
                        else:
                            signal = stat[algorithm]
                        if loud:
                            print(f"{algorithm}: {self.stats['TD_error'][algorithm]}")
                        if title == 'TD Error Per Time Slot':
                            if colors is not None:
                                plt.semilogy(list(range(0, len(stat[algorithm]))), signal, label=f'{algorithm}',c=colors[algorithm])
                            else:
                                plt.semilogy(list(range(0, len(stat[algorithm]))), signal, label=f'{algorithm}')
                        else:
                            if colors is not None:
                                plt.plot(list(range(0, len(stat[algorithm]))), signal, label=f'{algorithm}',c=colors[algorithm])
                            else:
                                plt.plot(list(range(0, len(stat[algorithm]))), signal, label=f'{algorithm}')
                    plt.title(title)
                    plt.xlabel(xlabel)
                    plt.ylabel(ylabel)
                    plt.legend()
                    plt.grid()
                    plt.savefig(self.result_dir + f"plots/{title}.png", dpi=1024)
                    plt.clf()
                except ValueError:
                    pass
        plt.clf()
        WINDOW_LENGTH = 1000
        WINDOW = np.ones(WINDOW_LENGTH) / WINDOW_LENGTH
        xlabels = ['Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep', 'Timestep']
        ylabels = [r'$\mathbb{E}[\delta_{TD}]$', r'${E}_{\pi}[\sum_{n=0}^{N-1}R^{(n)}_t]$', 'Delay', '#Packets', '#Packets', r'${E}_{\pi}[G_t-Q^{\pi}(S_t,A_t)]$']
        stats = [self.stats[exp] for exp in self.exp_names]
        titles = ['TD Error Per Time Slot', 'Rewards Per Time Slot', 'Average Delay Per Time Slot', 'Packet Drop Per Time Slot', 'Arrival Rate Per Time Slot', 'Return Vs Estimation Per Time Slot']
        withWindow = [True, True, True, True, True, True]
        plot_train_stats(stats, xlabels=xlabels, ylabels=ylabels, titles=titles, withWindows=withWindow)

class TestStatisticsLoader():
    def __init__(self, result_dir):
        self.exp_names = get_exp_names()
        self.ylabel_vec = get_ylabel_vec()
        # Initialize the stats dictionary
        self.stats = {exp: {} for exp in self.exp_names}
        self.load_vec = {}
        self.algorithms = []
        self.result_dir = result_dir

    def add_data(self, path, algorithm):
        currData = pd.read_pickle(path)
        self.load_vec[algorithm] = []
        self.algorithms.append(algorithm)
        # Set up the corresponding data structure
        for exp in self.exp_names:
            self.stats[exp][algorithm] = {}

        for load in currData["Load"][self.exp_names[0]].values:
            self.load_vec[algorithm].append(load)

        for exp in self.exp_names:
            for idx, load in enumerate(currData["Load"][exp].values):
                currLoadRes = currData["Results"][exp][idx]
                self.stats[exp][algorithm][load] = currLoadRes

    def append_data(self, path, algorithm, seed):
        if seed == 0:
            self.add_data(path, algorithm)
        else:
            currData = pd.read_pickle(path)
            for exp in self.exp_names:
                for idx, load in enumerate(currData["Load"][exp].values):
                    currLoadRes = currData["Results"][exp][idx]
                    self.stats[exp][algorithm][load] = np.hstack((self.stats[exp][algorithm][load], currLoadRes))

    def plot_data(self, loud, colors=None):
        mStyles = ["*", "h", "o", "v", ".", ",", "^", "<", ">", "1", "2", "3", "4", "8", "s", "p", "P", "H", "+", "x", "X", "D", "d", "|", "_", 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for exp, ylabel in zip(self.exp_names, self.ylabel_vec):
            if loud:
                print(f"{exp}")
            plt.clf()
            plt.title(f"{exp} vs Network Load")
            idx = 0
            for algorithm in self.algorithms:
                print(f'{exp}:{algorithm}')
                if loud:
                    print(f"{algorithm}: {self.stats[exp][algorithm]}")
                mean = np.array(list(map(lambda x: np.mean(x[1]), self.stats[exp][algorithm].items())))
                std = np.array(list(map(lambda x: np.std(x[1]), self.stats[exp][algorithm].items())))
                low_std = mean - std
                low_std[low_std < 0] = 0

                if exp == 'Arrival Ratio':
                    high_std = mean + std
                    high_std[high_std > 1] = 1
                else:
                    high_std = mean + std

                if colors is not None:
                    plt.plot(self.load_vec[algorithm], mean, c=colors[algorithm], label=algorithm, marker=mStyles[idx])
                    # plt.fill_between(self.load_vec[algorithm], low_std, high_std, alpha=0.25, facecolor=colors[algorithm])
                else:
                    plt.plot(self.load_vec[algorithm], mean, label=algorithm, marker=mStyles[idx])
                    # plt.fill_between(self.load_vec[algorithm], low_std, high_std, alpha=0.25)
                idx += 1

            plt.xlim([1.5, 6.5])
            plt.xlabel('Network Load [Packets]')
            plt.ylabel(ylabel)
            plt.legend(loc='best', prop={'size': 7}, ncol=2)
            plt.savefig(self.result_dir + f"plots/{exp}.png", dpi=500)
            plt.clf()

class MobilityTestStatisticsLoader():
    def __init__(self, result_dir):
        self.exp_names = get_exp_names()
        self.ylabel_vec = get_ylabel_vec()
        # Initialize the stats dictionary
        self.stats = {exp: {} for exp in self.exp_names}
        self.mobility_vec = {}
        self.algorithms = []
        self.result_dir = result_dir
        self.load_vec = np.arange(1,7,0.5)

    def add_data(self, paths, algorithm, mobility_vec):
        self.mobility_vec[algorithm] = mobility_vec
        self.algorithms.append(algorithm)
        # Set up the corresponding data structure
        for exp in self.exp_names:
            self.stats[exp][algorithm] = {}
        for mobility, path in zip(mobility_vec, paths):
            currData = pd.read_pickle(path)
            for exp in self.exp_names:
                self.stats[exp][algorithm][mobility] = {}
                for idx, load in enumerate(currData["Load"][exp].values):
                    currLoadRes = currData["Results"][exp][idx]
                    self.stats[exp][algorithm][mobility][load] = currLoadRes

    def append_data(self, paths, algorithm, seed, mobility_vec):
        if seed == 0:
            self.add_data(paths, algorithm, mobility_vec)
        else:
            for mobility, path in zip(mobility_vec, paths):
                currData = pd.read_pickle(path)
                for exp in self.exp_names:
                    for idx, load in enumerate(currData["Load"][exp].values):
                        currLoadRes = currData["Results"][exp][idx]
                        self.stats[exp][algorithm][mobility][load] = np.hstack((self.stats[exp][algorithm][mobility][load], currLoadRes))

    def plot_data(self, loud, colors=None):
        for load in self.load_vec:
            for exp, ylabel in zip(self.exp_names, self.ylabel_vec):
                results_list = []
                for algorithm in self.algorithms:
                    mean = list(map(lambda x: np.mean(x[1][load]), self.stats[exp][algorithm].items()))
                    results_list.append(mean)
                mobility_vec = self.mobility_vec[self.algorithms[0]]
                fig = plt.figure()
                ax = fig.add_axes([0, 0, 1, 1])
                X = np.arange(len(mobility_vec))
                for idx, algorithm in enumerate(self.algorithms):
                    ax.bar(X + 0.1*idx, results_list[idx], width=0.1,label=algorithm)
                ax.set_xticks(X)
                ax.set_xticklabels(mobility_vec)
                plt.legend(loc='best')
                plt.grid()
                plt.ylabel(ylabel)
                plt.xlabel('#Number-Of-Basestation')
                plt.savefig(self.result_dir + f"plots/{exp}_barplot_load_{load}.png",bbox_inches='tight')
                plt.clf()

class OnlineTestStatisticsCollector(object):
    def __init__(self, setting, result_dir, algorithms):
        self.exp_names = get_exp_names()
        self.ylabel_vec = get_ylabel_vec()

        self.maximal_timestep = setting["max_allowed_time_step_per_episode"]
        self.trials = setting["test_trials_per_load"]
        self.capacity = setting["capacity"]
        self.result_dir = result_dir
        self.algorithms = algorithms

        self.stats = {exp: {} for exp in self.exp_names}
        self.stats["Packet Drop Count"] = {}
        self.stats["Arrival Rate Trend"] = {}
        self.stats["Hop Count"] = {}
        self.stats["Circles Count"] = {}
        self.stats["Packet Drop Trend"] = {}
        self.stats["Load"] = {}
        self.prev_drop_historgram = {algorithm: np.zeros((self.trials, setting["num_bs"])) for algorithm in algorithms}
        for algorithm in algorithms:
            for exp in self.stats:
                self.stats[exp][algorithm] = np.zeros((self.trials, self.maximal_timestep))
            self.stats["Packet Drop Count"][algorithm] = np.zeros((self.trials, self.maximal_timestep))
            self.stats["Arrival Rate Trend"][algorithm] = np.zeros((self.trials, self.maximal_timestep))
            self.stats["Circles Count"][algorithm] = np.zeros((self.trials, self.maximal_timestep))
            self.stats["Hop Count"][algorithm] = np.zeros((self.trials, self.maximal_timestep))
            self.stats["Packet Drop Trend"][algorithm] = np.zeros((self.trials, self.maximal_timestep))
            self.stats["Load"][algorithm] = np.zeros((self.trials, self.maximal_timestep))
        self.curEpisode = {exp: 0 for exp in algorithms}
        self.TimeVec = np.arange(0, self.maximal_timestep)

    def add_stats(self, env, algorithm, currTrial, currTimeslot):
        try:
            self.stats["Average Delay Time"][algorithm][currTrial][currTimeslot] = sum(env.dynetwork._delivery_times[-100:]) / len(env.dynetwork._delivery_times[-100:])
        except ZeroDivisionError:
            self.stats["Average Delay Time"][algorithm][currTrial][currTimeslot] = 0

        self.stats["Maximum Number of Packets a Single Node Hold"][algorithm][currTrial][currTimeslot] = env.dynetwork._max_queue_length
        self.stats["Average Number of Packets a Single Node Hold"][algorithm][currTrial][currTimeslot] = env.dynetwork._avg_q_len_arr[-1]

        self.stats["Percentile of Working Nodes at Maximal Capacity"][algorithm][currTrial][currTimeslot] = (np.array(env.dynetwork._num_capacity_node) / np.array(env.dynetwork._num_working_node)).mean() * 100
        self.stats["Percentile of Non-Empty Nodes"][algorithm][currTrial][currTimeslot] = (((np.sum(env.dynetwork._num_working_node)) / env.nnodes)) * 100

        try:
            self.stats["Average Normalized Throughput"][algorithm][currTrial][currTimeslot] = sum(env.dynetwork._avg_throughput[-100:]) / len(env.dynetwork._avg_throughput[-100:])
        except ZeroDivisionError:
            self.stats["Average Normalized Throughput"][algorithm][currTrial][currTimeslot] = 0
        self.stats["Average Link Utilization"][algorithm][currTrial][currTimeslot] = np.mean(env.dynetwork._avg_link_utilization)
        try:
            self.stats["Average Transmission Rejections"][algorithm][currTrial][currTimeslot] = env.dynetwork._rejections / (env.dynetwork._initializations + env.npackets)
        except ZeroDivisionError:
            self.stats["Average Transmission Rejections"][algorithm][currTrial][currTimeslot] = 0
        try:
            self.stats["Arrival Ratio"][algorithm][currTrial][currTimeslot] = env.dynetwork._deliveries / (
                        env.dynetwork._generations + env.npackets)
        except ZeroDivisionError:
            self.stats["Arrival Ratio"][algorithm][currTrial][currTimeslot] = 0

        self.stats["Fairness"][algorithm][currTrial][currTimeslot] = np.nanmean(env.dynetwork._avg_router_fairness)
        delta_in_packet_drop = env.dropped_packet_histogram - self.prev_drop_historgram[algorithm][currTrial]
        self.stats["Packet Drop Percentile"][algorithm][currTrial][currTimeslot] = np.sum(delta_in_packet_drop)
        self.prev_drop_historgram[algorithm][currTrial] = env.dropped_packet_histogram
        self.stats["Arrival Rate Trend"][algorithm][currTrial][currTimeslot] = env.dynetwork._packet_arrival[-1]
        self.stats["Circles Count"][algorithm][currTrial][currTimeslot] = env.dynetwork._circles_counter[-1]
        try:
            self.stats["Hop Count"][algorithm][currTrial][currTimeslot] = env.dynetwork._hops_counter[-1]
        except IndexError:
            self.stats["Hop Count"][algorithm][currTrial][currTimeslot] = 0
        try:
            self.stats["Packet Drop Trend"][algorithm][currTrial][currTimeslot] = env.dynetwork._packet_drops[-1]
        except IndexError:
            self.stats["Packet Drop Trend"][algorithm][currTrial][currTimeslot] = 0

        self.stats["Load"][algorithm][currTrial][currTimeslot] = env.dynetwork._lambda_load

    def plot_result(self, loud=False, colors=None):
        WINDOW_LENGTH = 100
        WINDOW = np.ones(WINDOW_LENGTH) / WINDOW_LENGTH
        for algorithm in self.algorithms:
            for trial in range(self.trials):
                self.stats["Packet Drop Count"][algorithm][trial] = np.convolve(WINDOW,  self.stats["Packet Drop Percentile"][algorithm][trial], mode='same')
        self.ylabel_vec[-1] = "Packet Drop Count"
        self.exp_names[-1] = "Packet Drop Count"
        for exp, ylabel in zip(self.exp_names, self.ylabel_vec):
            if loud:
                print(f"{exp}")
            plt.clf()
            plt.title(f"{exp} vs Time-Slot")
            for algorithm in self.algorithms:
                mean = np.mean(self.stats[exp][algorithm],axis=0)
                std = np.std(self.stats[exp][algorithm],axis=0)
                low_std = mean - std
                low_std[low_std < 0] = 0
                if loud:
                    print(f"{algorithm}: {self.stats[exp][algorithm]}")
                if colors is not None:
                    plt.plot(self.TimeVec, np.convolve(WINDOW,  mean, mode='same'), c=colors[algorithm], label=algorithm)
                    plt.fill_between(self.TimeVec, np.convolve(WINDOW,  low_std, mode='same'), np.convolve(WINDOW,  mean + std, mode='same'), alpha=0.25, facecolor=colors[algorithm])
                else:
                    plt.plot(self.TimeVec,  np.convolve(WINDOW,  mean, mode='same'), label=algorithm)
                    plt.fill_between(self.TimeVec, np.convolve(WINDOW,  low_std, mode='same'), np.convolve(WINDOW,  mean+std, mode='same'), alpha=0.25)
            plt.xlabel('Time-Slot')
            plt.ylabel(ylabel)
            plt.xlim([200, 9975])
            plt.legend(loc='upper left',ncol=4,fontsize='xx-small')
            plt.grid()
            plt.savefig(self.result_dir + f"plots/{exp}.png", dpi=1024)
            plt.clf()

    def dump_statistics(self):
        for algorithm in self.algorithms:
            results_df = []
            for exp in self.exp_names:
                results_df.append(pd.DataFrame(list(self.stats[exp][algorithm])))
            final_res = pd.concat(results_df, axis=0, keys=self.exp_names)
            final_res.to_pickle(self.result_dir + f"data/Test_Statistic_{algorithm}.pkl")
            final_res.to_csv(self.result_dir + f"data/Test_Statistic_{algorithm}.csv")

    def add_data(self, path, algorithm):
        currData = pd.read_pickle(path).transpose()
        self.ylabel_vec[-1] = "Packet Drop Count"
        self.exp_names[-1] = "Packet Drop Count"

        if algorithm not in self.algorithms:
            self.algorithms.append(algorithm)

        # Set up the corresponding data structure
        invalid_exp = []
        for exp in self.exp_names:
            trials = currData[exp].shape[1]
            samples = currData[exp].shape[0]
            self.stats[exp][algorithm] = np.zeros((trials, samples))

        for exp in self.exp_names:
            if exp not in invalid_exp:
                for idx in currData[exp].keys():
                    currTrialRes = currData[exp][idx]
                    self.stats[exp][algorithm][idx] = currTrialRes

    def append_data(self, path, algorithm, seed):
        if seed == 0:
            self.add_data(path, algorithm)
        else:
            currData = pd.read_pickle(path).transpose()
            for exp in self.exp_names:
                tempRes = np.zeros((self.trials, self.maximal_timestep))
                for idx in currData[exp].keys():
                    tempRes[idx] = currData[exp][idx]
                self.stats[exp][algorithm] = np.vstack((self.stats[exp][algorithm], tempRes))

    def plot_multiple_result(self, loud=False, colors=None):
        WINDOW_LENGTH = 100
        WINDOW = np.ones(WINDOW_LENGTH) / WINDOW_LENGTH
        for algorithm in self.algorithms:
            for trial in range(self.stats["Packet Drop Percentile"][algorithm].shape[0]):
                self.stats["Packet Drop Count"][algorithm][trial] = np.convolve(WINDOW,  self.stats["Packet Drop Percentile"][algorithm][trial], mode='same')
        self.ylabel_vec[-1] = "Packet Drop Count"
        self.exp_names[-1] = "Packet Drop Count"
        for exp, ylabel in zip(self.exp_names, self.ylabel_vec):
            if loud:
                print(f"{exp}")
            plt.clf()
            if exp == 'Packet Drop Trend':
                plt.title(f"Dropped Packets vs Time-Slot")
            else:
                plt.title(f"{exp} vs Time-Slot")
            for algorithm in self.algorithms:
                mean = np.mean(self.stats[exp][algorithm],axis=0)
                std = np.std(self.stats[exp][algorithm],axis=0)
                low_std = mean - std
                low_std[low_std < 0] = 0
                if loud:
                    print(f"{algorithm}: {self.stats[exp][algorithm]}")
                if colors is not None:
                    plt.plot(self.TimeVec, np.convolve(WINDOW,  mean, mode='same'), c=colors[algorithm], label=algorithm)
                    # plt.fill_between(self.TimeVec, np.convolve(WINDOW,  low_std, mode='same'), np.convolve(WINDOW,  mean + std, mode='same'), alpha=0.25, facecolor=colors[algorithm])
                else:
                    plt.plot(self.TimeVec,  np.convolve(WINDOW,  mean, mode='same'), label=algorithm)
                    # plt.fill_between(self.TimeVec, np.convolve(WINDOW,  low_std, mode='same'), np.convolve(WINDOW,  mean+std, mode='same'),alpha=0.25)
            plt.xlabel('Time-Slot')
            plt.ylabel(ylabel)
            plt.xlim([400, 9950])
            # plt.ylim([11,24])
            plt.legend(loc="best", ncol=2, prop={'size': 6})
            plt.grid()
            plt.savefig(self.result_dir + f"plots/{exp}.png", dpi=1024)
            plt.clf()























