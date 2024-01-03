import pickle

import numpy as np

DATA_FILENAME = 'greedy_scores.p'


def table_printer() -> None:
    """Prints design sequences and rewards at desired ranks."""  # TODO reword
    scores = pickle.load(open(DATA_FILENAME, "rb"))
    score_mat = np.zeros((10, 10))
    for i, the_list in enumerate(scores):
        for j, element in enumerate(the_list):
            score_mat[j, i] = "{:.2f}".format(element)
    print(np.flip(score_mat, axis=0))


if __name__ == "__main__":
    table_printer()