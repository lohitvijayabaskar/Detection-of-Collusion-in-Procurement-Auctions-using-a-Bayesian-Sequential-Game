# Detection of Collusion in Procurement Auctions using a Bayesian Sequential Game

This project explores the intersection of **Multi-Agent Reinforcement Learning (MARL)** and **Mechanism Design**, specifically focusing on how autonomous bidding agents adapt to—and attempt to evade—automated regulatory oversight in procurement auctions through dynamic cartel formation and discrete strategy selection.

## Overview

In this simulated closed-loop environment, 20 autonomous firms (RL agents) compete in a repeated procurement auction. The firms are trained using Proximal Policy Optimization (PPO) to maximize their profit.

Simultaneously, a **Bayesian Sequential Regulator** actively monitors the auction without access to the firms' private costs. It evaluates the public bids using statistical screens (Coefficient of Variation, Normalized Difference, Skewness) and runs sequential change-point detection algorithms (**CUSUM** and **Shiryaev-Roberts**) to calculate the posterior probability of a cartel `P(cartel)`. 

If the regulator detects anomalous bidding patterns indicative of collusion, it triggers an alarm and levies severe fines on the suspicious firms.

## Implementation Details & Technicalities

### 1. Multi-Discrete Action Spaces
Firms do not simply choose a continuous markup. Instead, they operate in a `MultiDiscrete([6, 2, 2])` action space representing:
- **Bidding Strategy (0-5)**: The firm's choice of economic bidding behaviour.
- **Admission Vote (0-1)**: The firm's vote on how the cartel should admit new members.
- **Punishment Vote (0-1)**: The firm's vote on how the cartel should punish defectors.

### 2. Firm Strategies
Firms can choose between several honest and dishonest economic strategies every round:
- **0: Honest Bertrand:** Bids aggressively near actual cost to undercut competitors.
- **1: Honest Cournot (Low):** Bids with a low, competitive markup.
- **2: Honest Cournot (High):** Bids with a moderate markup.
- **3: Honest Random:** Bids with highly variable noise, introducing unpredictability.
- **4: Dishonest Target Price:** Bids near the regulator's reserve price (maximally collusive).
- **5: Dishonest Cover Bid:** If the firm is the "designated winner" of the cartel, it bids near cost to secure the win. Otherwise, it intentionally bids high to create a "cover" for the designated winner.

### 3. Cartel Mechanics
Cartel behaviour is not scripted top-down; it is dynamically managed by a `Cartel` class that processes the firms' votes to determine collective policies:
- **Voting Mechanism**: Each round, current cartel members cast votes via their action space. The majority vote determines the active Admission and Punishment policies for the round.
- **Admission Policies**: Firms employing dishonest strategies are treated as applying for the cartel. The cartel may admit them unconditionally (Open Admission) or only if their cost is below the market average (Selective Admission).
- **Punishment Policies**: If a cartel member undercuts the designated winner (defecting), they are punished based on the voted policy. Punishments range from a temporary 1-round ban (Tit-for-Tat) to a permanent ban (Grim Trigger).
- **Designated Winner**: The cartel designates the member with the lowest private cost as the winner for the current round, allowing them to optimize collective profit.

### 4. Co-adaptation Loop
Because the regulator operates *in-the-loop* (agents observe the regulator's suspicion levels), the RL agents learn to co-adapt. When competitive margins are squeezed to zero, the agents dynamically learn to balance between selecting Dishonest strategies to maximize profit and Honest strategies to avoid regulatory fines.

## Files
- `emergent_procurement_env.py`: The PettingZoo ParallelEnv simulating the auction, firm strategies, cartel mechanics, and enforcing the RL observation/action spaces.
- `regulator.py`: The Bayesian sequential change-point detection algorithm.
- `train_emergent_rl.py`: The Ray RLlib PPO training loop. Logs the distribution of chosen strategies to `coadaptation_log.csv`.
- `eval_emergent.py`: The evaluation script to observe the converged policies and regulator alarms in action.

## Running the Code
1. Install dependencies: `pip install -r requirements.txt`
2. Train the agents (100 iterations): `python train_emergent_rl.py`
3. Evaluate the equilibrium: `python eval_emergent.py`
