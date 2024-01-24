# Dynamic Programming

This directory details the algorithm in sections **III-C** and **IV-C** of the 
paper.

`experiment.py` generates a `dp_values.p` file that contains all values 
of each design sequence of nonzero length found by DP. This data is used for 
generating the table and figure. It is also used for the ADP experiments 
when ground truth values are needed. This code runs in a few minutes on our 
machine.

`table.py` prints values of desired design sequences, which can be 
specified in the file. 

`figure.py` saves a ridgeline plot `dp_results.png` of the value functions 
after each experiment.