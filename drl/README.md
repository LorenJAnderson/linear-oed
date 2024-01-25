# Deep Reinforcement Learning

This directory details the algorithm in sections **III-E** and **IV-E** of the 
paper.

`experiment.py` generates a `trajectories.p` file of 100_000 random 
trajectories that include the final design sequence and the associated reward.
It then uses the random trajectory data to learn state-action regressors 
for each experiment, and the learned regressors are stored in a `regressors.p`
file. This code takes roughly half an hour to run on our machine. 

`table.py` prints the current design sequence, the most promising action 
given the current design sequence, and the expected score by continually 
choosing the most promising action. 

`figure.py` saves a violin plot `adp_results.png` of the error in the 
estimated and true state-action values in the regressors for all design 
sequences, conditioned on the experiment number. It uses the
`../dp/dp_values.p` dynamic programming file for ground truth values of each 
design sequence. This code takes roughly one minute to run.