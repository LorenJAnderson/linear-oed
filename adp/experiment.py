import formulas
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import numpy as np
import pickle
from itertools import combinations_with_replacement as cwr


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]


def create_targets9(states):
    targets = []
    for i, state in enumerate(states):
        if (i%10000) == 0:
            print(i)
        scores = []
        for des in DESIGN_SPACE:
            mean = np.array([[0.0], [0.0]])
            prior_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
            obs = tuple([1.0] * 10)
            des_seq = tuple(sorted(state[0:9] + [des]))
            _, post_cov = formulas.batch_update(mean, prior_cov, des_seq, obs, sigma=1.0)
            scores.append(formulas.calc_exp_kl(post_cov, prior_cov))
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
print(len(states))
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

regs = pickle.load(open('regressors.p', "rb"))
des_seq = []
for i in range(1, 11):
    reg = regs[i]
    top_des = 0.1
    top_score = 0
    for des in DESIGN_SPACE:
        new_des_seq = des_seq + [des]
        score = reg.predict([new_des_seq])[0][0]
        if score > top_score:
            top_des = des
            top_score = score
    des_seq.append(top_des)
    print(des_seq, top_score)
