# Deep Reinforcement Learning

This directory details the algorithm in sections **III-E** and **IV-E** of the 
paper.

`experiment.py` creates a `logs` folder and generates two files that detail 
the training and testing progress of a DQN agent applied to the linear OED 
environment. The first is a TensorBoard event file containing the training 
rewards, and the second is an `evaluations.npz` file that contains the 
testing rewards. The event file will need to be renamed when generating the 
table and figure. This code runs in a few minutes on our machine.

`table.py` prints the DQN training and testing reward values.

`figure.py` saves a line plot `drl_results.png` of the DQN training and 
testing rewards.