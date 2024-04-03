import torch
import functools
import numpy as np
import matplotlib.pyplot as plt

from model import CoronaryNet
from data import load_test


def evaluate_qualitative(model: CoronaryNet, ix):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_test, loader_test = load_test(100)

    x, gt = data_test[ix]  
    #pred = model(x)

    data_flat = functools.reduce(lambda a, b: np.concatenate((a, b)), gt[1:], gt[0])    
    data_flat = data_flat[::10]


    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        [x for (x, _, _) in data_flat],
        [y for (_, y, _) in data_flat],
        [z for (_, _, z) in data_flat],
        marker='o'
        )

    ax.scatter(0, 0, 0, marker="s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":

    model = CoronaryNet()
    #model.load_state_dict(torch.load("models/weights"))
    evaluate_qualitative(
        model,
        ix=0
    )