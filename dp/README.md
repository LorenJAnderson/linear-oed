# Dynamic Programming

This directory details the algorithm in sections **III-C** and **IV-C** of the 
paper.

`experiment.py` generates a `dp_values.p` file that contains all the values of each design sequence found by DP. This data is used for generating the table and figure It is also used for the ADP... This code runs in under one minute.  # TODO time and ADP

`table.py` prints values of desired design sequences, which can be specified in the file. 

`figure.py` saves a ridgeline plot `dp_results.png` of the value functions after each experiment.