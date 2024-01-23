# Batch Design

This directory details the algorithm in sections **III-A** and **IV-A** of the 
paper.

`experiment.py` generates a `batch_scores.p` file that contains all batch 
design sequences of length 10 and corresponding scores. This data is used 
for generating the table and figure. This code runs in under one minute on 
our machine.

`table.py` prints a table of design sequence scores of desired ranks, which 
can be specified in the file. 

`figure.py` saves a histogram `batch_results.png` of the scores.