# Routing-In-IAB-Networks
This repository contains the implementation details of the paper - 
Multi-Agent Reinforcement Learning for Network-Routing in Integrated Access Backhaul Networks

In this study, we examine the problem of wireless routing in IAB networks, involving fiber-connected base stations, wireless base stations, and multiple users. Physical constraints limit the use of a central controller in these networks, so base stations have limited access to the network status. This network operates in a time-slotted regime, where base stations monitor network conditions and forward packets accordingly.
Our objective is to maximize the arrival ratio of packets while simultaneously minimizing their latency. To accomplish this, we formulate this problem as a multi-agent partially observed Markov decision process. Furthermore, we develop an algorithm that uses Multi-Agent Reinforcement Learning combined with Advantage Actor Critic (A2C) to derive a joint routing policy on a distributed basis. Due to the importance of packet destinations for successful routing decisions, we utilize information about similar destinations as a basis for selecting specific-destination routing decisions. For portraying the similarity between those destinations, we rely on their relational base-station associations, i.e., which base station they are currently connected to. Therefore, the algorithm is referred to as Relational Advantage Actor Critic (Relational A2C). To the best of our knowledge, this is the first work that optimizes routing strategy for IAB networks. Further, we present three types of training paradigms for this algorithm in order to provide flexibility in terms of its performance and throughput. Through numerical experiments of different network scenarios, Relational A2C algorithms were demonstrated to be capable of achieving near-centralized performance even though they operate in a decentralized manner in the network of interest. With those experiments, we compare Relational A2C to other reinforcement learning algorithms, like Q-Routing and Hybrid Routing. This comparison illustrates that solving the joint optimization problem increases network efficiency and reduces selfish agent behavior.

![Problem Diagram](https://user-images.githubusercontent.com/49431639/207116476-c21eec37-8604-4fb9-a499-d32c24fb8f04.png)

For this research details we refer the reader to read our paper.

#Citation

If you find either the code or the paper useful for your research, cite our paper:

