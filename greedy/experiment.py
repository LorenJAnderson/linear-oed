import formulas
import numpy as np
import pickle


DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DES_SEQ_LEN = 10


def det_best_design(mean, cov, obs, sig):
    best_score = -1
    best_design = 0
    scores = []
    for des in DESIGN_SPACE:
        post_mean, post_cov = formulas.batch_update(mean, cov, [des], obs, sig)
        score = formulas.calc_exp_kl(post_cov, cov)
        scores.append(score)
        if score > best_score:
            best_score = score
            best_design = des
    post_mean, post_cov = formulas.batch_update(mean, cov, [best_design], obs, sig)
    return best_design, best_score, post_mean, post_cov, scores


def experiment():
    greedy_des_seq = []
    all_scores = []
    mean = np.array([[0], [0]])
    cov = np.array([[1, 0], [0, 1]])
    for _ in range(DES_SEQ_LEN):
        design, score, mean, cov, scores = det_best_design(mean, cov, 1, 1)
        greedy_des_seq = greedy_des_seq + [design]
        all_scores.append(scores)
    print(greedy_des_seq)
    print(all_scores)
    filename = 'data.p'
    pickle.dump(all_scores, open(filename, "wb"))


if __name__ == "__main__":
    experiment()
