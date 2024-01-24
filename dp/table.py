import pickle

DES_SPACE = [round(0.1 * i, 1) for i in range(1, 11)]
DATA_FILENAME = 'dp_values.p'


def desired_seq_printer() -> None:
    """Prints desired design sequences and corresponding values found by DP."""
    value_dict = pickle.load(open(DATA_FILENAME, "rb"))
    desired_sequences = [key for key in value_dict.keys() if len(key) == 1]
    for key in desired_sequences:
        print(key, value_dict[key])


if __name__ == "__main__":
    desired_seq_printer()

