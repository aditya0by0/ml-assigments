import numpy as np
import pandas as pd
import sys


def linear_regression(data, eta, threshold):
    data_df = pd.read_csv(data, header=None)
    data_df.insert(0, 'bias', 1)
    X_df = data_df.iloc[:, :-1]
    y_df = data_df.iloc[:, -1]

    weights = np.zeros(X_df.shape[1])

    iter_, prev_error = 0, 0
    while True:
        
        y_pred = np.dot(X_df, weights)
        errors = y_df - y_pred
        sse = np.sum(np.square(errors))
        print(f"{iter_},{','.join(map(str, weights))},{sse}")
        
        gradient = np.dot(X_df.T, errors)
        weights += eta * gradient

        if np.absolute(sse - prev_error) < threshold:
            break

        iter_ += 1
        prev_error = sse


if __name__ == "__main__":
    data = sys.argv[2]
    eta = float(sys.argv[4])
    threshold = float(sys.argv[6])

    linear_regression(data, eta, threshold)
