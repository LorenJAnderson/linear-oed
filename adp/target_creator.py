import formulas
import numpy as np
import pickle
import copy

DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]

states = [[0.1, 0.1, 0.1, 0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]]
for i in range(10000):
    episode_designs = []
    for j in range(10):
        idx = np.random.randint(10)
        design = DESIGN_SPACE[idx]
        episode_designs.append(design)
    states.append(episode_designs)

scores = [formulas.big_kl_div(state) for state in states]
print(np.max(scores), states[np.argmax(scores)])


filename = 'data.p'
pickle.dump(states, open(filename, "wb"))