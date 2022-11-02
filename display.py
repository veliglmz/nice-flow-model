import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")


def save_scatter_plot(d0: np.ndarray, d1: np.ndarray, output_path: str="output.png" ):
    x0 = d0[:, 0]
    x1 = d0[:, 1]
    x0_hat = d1[:, 0]
    x1_hat = d1[:, 1]
    fig, ax = plt.subplots()
    ax.scatter(x0, x1, c="blue", label="train", alpha=0.5)
    ax.scatter(x0_hat, x1_hat, c="red", label="test", alpha=0.5)
    ax.set_xlabel("x0")
    ax.set_ylabel("x1")
    plt.legend(loc="upper left")
    plt.savefig(output_path)   