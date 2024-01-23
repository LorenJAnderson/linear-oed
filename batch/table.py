import pickle

DATA_FILENAME = 'batch_scores.p'
DESIRED_RANKS = [0, 1, 2, 3, 4, 5, 10, 100, 200, 500, 1000, 2000, 5000,
                 10000, 20000, 50000, 92377]


def table_printer() -> None:
    """Prints design sequences and rewards at desired ranks."""
    scores = pickle.load(open(DATA_FILENAME, "rb"))
    scores_and_des_seqs = [(scores[key], key) for key in list(scores)]
    sorted_scores_and_des_seqs = sorted(scores_and_des_seqs, reverse=True)

    for rank, (score, des_seq) in enumerate(sorted_scores_and_des_seqs):
        if rank in DESIRED_RANKS:
            des_seq = tuple([round(des, 1) for des in list(des_seq)])
            print(str(rank) + " " + str(des_seq) +
                  " " + '{:.3f}'.format(score))


if __name__ == "__main__":
    table_printer()
