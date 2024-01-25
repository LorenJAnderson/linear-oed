# linear-oed
Code for paper *Bayesian Sequential Optimal Experimental Design for Linear 
Regression with Reinforcement Learning*, doi: 10.1109/ICMLA55696.2022.00106.

---

# Installation
The code includes a `requirements.txt` file for package installation.

---

# Notes
 - Unless otherwise specified in the individual READMEs, the code runs 
instantaneously. The longest script takes roughly half an hour to run.
 - After running `drl/experiment.py`, the TensorBoard event file will need 
   to be renamed when analyzing the data in the `drl/table.py` and 
   `drl/figure.py` files.  