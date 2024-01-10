# Greedy Design

This directory details the algorithm in sections **III-B** and **IV-B** of the 
paper.

`experiment.py` generates a `greedy_scores.p` file that contains marginal scores from the greedy trajectory. This data is used for generating the data table and figure. This code runs nearly instantaneously.

`table.py` prints a table of marginal scores for each design for each experiment, assuming that the previous design sequence was chosen greedily.

`figure.py` saves a histogram `greedy_figure.png` of the marginal scores.