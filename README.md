# Detection of Collusion in Procurement Auctions using a Bayesian Sequential Game

This project explores the intersection of **Multi-Agent Reinforcement Learning (MARL)** and **Mechanism Design**, specifically focusing on how autonomous bidding agents adapt to—and attempt to evade—automated regulatory oversight in procurement auctions.

## Overview

In this simulated closed-loop environment, 20 autonomous firms (RL agents) compete in a repeated procurement auction. The firms are trained using Proximal Policy Optimization (PPO) to maximize their profit.

Simultaneously, a **Bayesian Sequential Regulator** actively monitors the auction without access to the firms' private costs. It evaluates the public bids using statistical screens (Coefficient of Variation, Normalized Difference, Skewness) and runs sequential change-point detection algorithms (**CUSUM** and **Shiryaev-Roberts**) to calculate the posterior probability of a cartel `P(cartel)`. 

If the regulator detects anomalous bidding patterns indicative of collusion (such as complementary/cover bidding), it triggers an alarm and levies severe fines on the suspicious firms.

## Emergent Behavior
Because the regulator operates *in-the-loop* (agents observe the regulator's suspicion levels), the RL agents learn to co-adapt. When competitive margins are squeezed to zero, the agents dynamically abandon fair competition and organically learn textbook **Cover Bidding** strategies (where one agent bids low to win, while others submit intentionally inflated fake bids) in an attempt to manipulate the market, resulting in a fascinating adversarial cat-and-mouse dynamic.

## Files
- `emergent_procurement_env.py`: The PettingZoo ParallelEnv simulating the auction and enforcing the RL observation/action spaces.
- `regulator.py`: The Bayesian sequential change-point detection algorithm.
- `train_emergent_rl.py`: The Ray RLlib PPO training loop.
- `eval_emergent.py`: The evaluation script to observe the converged policies and regulator alarms in action.

## Running the Code
1. Install dependencies: `pip install -r requirements.txt`
2. Train the agents: `python train_emergent_rl.py`
3. Evaluate the equilibrium: `python eval_emergent.py`
