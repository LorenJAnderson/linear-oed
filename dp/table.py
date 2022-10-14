import pickle


if __name__ == "__main__":
    all_sequences = pickle.load(open("data.p", "rb"))
    for key in all_sequences.keys():
        if len(key) == 1:
            print(key, all_sequences[key])