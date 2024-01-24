import pickle

DATA_FILENAME = 'regressors.p'
DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
TOT_EXPS = 10

def table_printer() -> None:
    """Prints state, action, and estimated state action value of most promising
     action for each experiment. """
    regressors = pickle.load(open(DATA_FILENAME, "rb"))
    des_seq = []
    for exp in range(TOT_EXPS):
        regressor = regressors[exp]
        scores = [regressor.predict([sorted(des_seq + [des])])[0][0]
                  for des in DESIGN_SPACE]
        scores_and_designs = zip(scores, DESIGN_SPACE)
        best_score, best_des = sorted(scores_and_designs, reverse=True)[0]
        print(des_seq, best_des, best_score)
        des_seq.append(best_des)


if __name__ == "__main__":
    table_printer()
