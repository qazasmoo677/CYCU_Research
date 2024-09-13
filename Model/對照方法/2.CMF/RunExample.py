import matplotlib.pyplot as plt
import numpy as np
from LoadData import load_rating_data
from sklearn.model_selection import train_test_split
from ProbabilisticMatrixFactorization import PMF

if __name__ == "__main__":
    pmf = PMF()
    pmf.set_params({"num_feat": 10, "epsilon": 1, "_lambda": 0.1, "momentum": 0.8, "maxepoch": 10, "num_batches": 100,
                    "batch_size": 1000})
    ratings = load_rating_data()
    train, test = train_test_split(ratings, test_size=0.2)
    pmf.fit(train, test)