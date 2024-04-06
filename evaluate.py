import torch
import torch.nn as nn
import functools
import numpy as np
import matplotlib.pyplot as plt

from model import CoronaryNet
from data import load_test, load_train


def evaluate_qualitative(model: CoronaryNet, ix, device, data_test=load_test(100)[0], verbose=True):
    model.to(device)
    model.eval()
    x, x_acq, gt = data_test[ix]  

    x = x.to(device, dtype=torch.float)
    x = x.expand(1, -1, -1, -1)
    x_acq = x_acq.expand(1, -1, -1)
    pred = model(x, x_acq)
    pred = pred.reshape(model.M * model.N, 3)

    output = pred.reshape(model.M, model.N, 3)
    loss = nn.MSELoss()(output, torch.Tensor(gt))
    if verbose: print("mse:", loss)

    data_flat = functools.reduce(lambda a, b: np.concatenate((a, b)), gt[1:], gt[0])    

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.scatter(
        [x for (x, _, _) in data_flat],
        [y for (_, y, _) in data_flat],
        [z for (_, _, z) in data_flat],
        marker='o'
        )

    pred = pred.detach().numpy()

    ax.scatter(
        [x for (x, _, _) in pred],
        [y for (_, y, _) in pred],
        [z for (_, _, z) in pred],
        marker='x'
    )

    ax.scatter(0, 0, 0, marker="s")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()


if __name__ == "__main__":

    device = torch.device("cpu")
    model = torch.load("models/weights")
    data_test, _ = load_train(100)

    evaluate_qualitative(
        model,
        ix=6,
        device=device,
        data_test=data_test
    )