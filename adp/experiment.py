import pickle

import numpy as np
from sklearn.neighbors import KNeighborsRegressor

import formulas


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
NUM_TRAJECTORIES = 100_000
TRAJECTORY_DATA_FILENAME = 'trajectories.p'
REGRESSOR_DATA_FILENAME = 'regressors.p'


def generate_trajectories() -> None:
    """Generates random trajectories and stores each terminal state and associated
    reward in a list. Dumps list into pickle file."""
    all_trajectories = []
    for _ in range(NUM_TRAJECTORIES):
        state = []
        for _ in range(10):
            state.append(np.random.choice(DESIGN_SPACE))
        mean = np.array([[0.0], [0.0]])
        prior_cov = np.array([[1.0, 0.0], [0.0, 1.0]])
        obs = tuple([1.0] * 10)
        _, post_cov = formulas.batch_update(mean, prior_cov, tuple(state), obs, sigma=1.0)
        reward = formulas.calc_exp_kl(post_cov, prior_cov)
        all_trajectories.append((state, reward))
    pickle.dump(all_trajectories, open(TRAJECTORY_DATA_FILENAME, "wb"))


def create_targets(trajectories, exp, regressor=None):
    """Creates the target action-values at given experiment. Terminal reward is used as
    the target during the last experiment. Otherwise, the maximum action-value of the
    next state is used as the target.

    Keyword arguments:
    trajectories -- list of (terminal state, reward) trajectory tuples
    exp -- experiment number
    regressor --  current regressor
    """
    targets = []
    for state, reward in trajectories:
        if exp == 9:
            targets.append(reward)
        else:
            scores = [regressor.predict([sorted(state[0:exp+1] + [des])])
                                         for des in DESIGN_SPACE]
            targets.append(max(scores))
    return np.array(targets)


def experiment() -> None:
    """
    Fits action-value regressor for each experiment. Dumps dictionary of regressors
    into pickle file.
    """
    regressor_dict = {}
    trajectories = pickle.load(open(TRAJECTORY_DATA_FILENAME, "rb"))
    for exp in range(9, -1, -1):
        regressor = KNeighborsRegressor(n_neighbors=5)
        current_reg = None
        if exp != 9:
            current_reg = regressor_dict[exp+1]
        Y = create_targets(trajectories, exp, current_reg)
        X = np.array([sorted(state[0:exp+1]) for state, _ in trajectories])
        Y = Y.reshape(-1, 1)
        print(X.shape, Y.shape)
        regressor.fit(X=X, y=Y)
        print(np.max(regressor.predict(X)))
        regressor_dict[exp] = regressor
    pickle.dump(regressor_dict, open(REGRESSOR_DATA_FILENAME, "wb"))


if __name__ == "__main__":
    generate_trajectories()
    experiment()
