import pickle

DESIGN_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
regressors = pickle.load(open("regressors.p", "rb"))

des_seq = []
for i in range(0, 10):
    best_score = -100
    best_des = 0
    for des in DESIGN_SPACE:
        new_des_seq = sorted(des_seq + [des])
        score = regressors[i].predict([new_des_seq])
        if score > best_score:
            best_des = des
            best_score = score
    print(des_seq, best_des, best_score)
    des_seq.append(best_des)

