import pickle
from itertools import combinations_with_replacement as cwr

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

import formulas


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
NUM_TRAJECTORIES = 10_000  # TODO Change back


def generate_trajectories() -> None:
    all_trajectories = []
    for _ in range(NUM_TRAJECTORIES):
        trajectory = []
        for _ in range(10):
            trajectory.append(np.random.choice(DESIGN_SPACE))
        mean = np.array([[0.0], [0.0]])
        prior_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        obs = tuple([1.0] * 10)
        _, post_cov = formulas.batch_update(mean, prior_cov, tuple(trajectory),
                                            obs, sigma=1.0)
        reward = formulas.calc_exp_kl(post_cov, prior_cov)
        all_trajectories.append((trajectory, reward))
    pickle.dump(all_trajectories, open('trajectories.p', "wb"))


def create_targets(trajectories, exp, reg):
    targets = []
    for state, reward in trajectories:
        if exp == 10:
            targets.append(reward)
        else:
            scores = [reg.predict([sorted(state[0:exp] + [des])]) for des in DESIGN_SPACE]
            targets.append(max(scores))
    return np.array(targets)


def experiment() -> None:
    regs = {i: None for i in range(1, 11)}
    trajectories = pickle.load(open('trajectories.p', "rb"))
    for exp in range(10, 0, -1):
        reg = KNeighborsRegressor(n_neighbors=5)
        current_reg = regs[exp]
        if exp != 10:
            current_reg = regs[exp+1]
        Y = create_targets(trajectories, exp, current_reg)
        X = np.array([sorted(state[0:exp]) for state, _ in trajectories])
        Y = Y.reshape(-1, 1)
        print(X.shape, Y.shape)
        reg.fit(X=X, y=Y)
        print(np.max(reg.predict(X)))
        regs[exp] = reg
    pickle.dump(regs, open('regressors.p', "wb"))


if __name__ == "__main__":
    # generate_trajectories()
    experiment()
