import pickle


def top_n_designs(design_seq_values, top_n):
    """Prints design sequences, rewards, and ranks from ranks in given list"""
    sorted_vals = sorted([(design_seq_values[key], key) for key
                         in list(design_seq_values)], reverse=True)
    for val in top_n:
        print(str(val) + " & " + str(sorted_vals[val][1]) + " & " +
              '{:.3f}'.format(sorted_vals[val][0]) + " \\\\")


if __name__ == "__main__":
    design_seq_values = pickle.load(open("data.p", "rb"))
    top_n_designs(design_seq_values, list(range(10)))
    sorted_vals = sorted([(design_seq_values[key], key) for key
                          in list(design_seq_values)], reverse=True)
    print(sorted_vals[-1])