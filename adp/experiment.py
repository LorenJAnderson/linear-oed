import formulas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from itertools import combinations_with_replacement as cwr


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]


def create_targets9(states):
    targets = []
    for state in states:
        scores = [formulas.big_kl_div(sorted(state[0:9] + [des])) for des in DESIGN_SPACE]
        targets.append(max(scores))
    return np.array(targets)


def create_targets(states, exp, reg):
    targets = []
    for state in states:
        scores = [reg.predict([sorted(state[0:exp] + [des])]) for des in DESIGN_SPACE]
        targets.append(max(scores))
    return np.array(targets)


regs = {i: None for i in range(0, 10)}
# states = pickle.load(open("data.p", "rb"))
states = list(cwr(DESIGN_SPACE, 10))
new_states = []
for state in states:
    new_states.append(list(state))
states = new_states
# print(states)
for i in range(10):
    print(10-i)
    exp = 10-i
    reg = KNeighborsRegressor(n_neighbors=5)
    if exp == 10:
        Y = create_targets9(states)
    else:
        Y = create_targets(states, exp, regs[exp+1])
    X = np.array([sorted(state[0:exp]) for state in states])
    Y = Y.reshape(-1, 1)
    print(X.shape, Y.shape)
    reg.fit(X=X, y=Y)
    print(np.max(reg.predict(X)))
    regs[exp] = reg
pickle.dump(regs, open('regressors.p', "wb"))