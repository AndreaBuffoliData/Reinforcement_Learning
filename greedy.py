#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 15:05:14 2024

@author: andreabuffoli
"""

import numpy as np
import matplotlib.pyplot as plt

# Number of bandit arms
k = 10
# Number of steps
steps = 1000
# Number of runs to average over
runs = 2000

def bandit_problem():
    # True action values are normally distributed with mean 0 and variance 1
    true_action_values = np.random.randn(k)
    return true_action_values

def get_reward(true_value):
    # Reward for each action is normally distributed with the true value as mean and variance 1
    return np.random.randn() + true_value

def epsilon_greedy(epsilon, steps, runs):
    rewards = np.zeros((runs, steps))
    optimal_action_counts = np.zeros((runs, steps))
    
    for run in range(runs):
        true_action_values = bandit_problem()
        optimal_action = np.argmax(true_action_values)
        estimated_action_values = np.zeros(k)
        action_counts = np.zeros(k)
        
        for step in range(steps):
            if np.random.rand() < epsilon:
                action = np.random.choice(k)
            else:
                action = np.argmax(estimated_action_values)
            
            reward = get_reward(true_action_values[action])
            action_counts[action] += 1
            estimated_action_values[action] += (reward - estimated_action_values[action]) / action_counts[action]
            
            rewards[run, step] = reward
            if action == optimal_action:
                optimal_action_counts[run, step] = 1
                
    average_rewards = np.mean(rewards, axis=0)
    optimal_action_perc = np.mean(optimal_action_counts, axis=0) * 100
    
    return average_rewards, optimal_action_perc

epsilons = [0, 0.01, 0.1]
average_rewards = []
optimal_action_percs = []

for epsilon in epsilons:
    avg_reward, opt_action_perc = epsilon_greedy(epsilon, steps, runs)
    average_rewards.append(avg_reward)
    optimal_action_percs.append(opt_action_perc)

# Plotting the results
plt.figure(figsize=(14, 7))

# Average reward plot
plt.subplot(2, 1, 1)
for epsilon, avg_reward in zip(epsilons, average_rewards):
    plt.plot(avg_reward, label=f'$\epsilon$={epsilon}')
plt.xlabel('Steps')
plt.ylabel('Average reward')
plt.legend()

# Optimal action percentage plot
plt.subplot(2, 1, 2)
for epsilon, opt_action_perc in zip(epsilons, optimal_action_percs):
    plt.plot(opt_action_perc, label=f'$\epsilon$={epsilon}')
plt.xlabel('Steps')
plt.ylabel('% Optimal action')
plt.legend()

plt.show()

