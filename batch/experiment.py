import formulas
import numpy as np
from itertools import combinations_with_replacement as cwr
import pickle


DESIGN_SPACE = [0.1 * i for i in range(1, 11)]
DES_SEQ_LEN = 10
DESIGN_SEQUENCES = list(cwr(DESIGN_SPACE, DES_SEQ_LEN))


def experiment():
    results = {des: None for des in DESIGN_SEQUENCES}
    for i, des_seq in enumerate(DESIGN_SEQUENCES):
        if (i % 10000) == 0:
            print(i, ' out of ', len(DESIGN_SEQUENCES))
        mean = np.array([[0], [0]])
        cov = np.array([[1, 0], [0, 1]])
        post_mean, post_cov = formulas.batch_update(mean, cov, des_seq, 1, 1)
        score = formulas.calc_exp_kl(post_cov, cov)
        results[des_seq] = score
    filename = 'data.p'
    pickle.dump(results, open(filename, "wb"))


if __name__ == "__main__":
    experiment()