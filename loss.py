import torch


def arc_loss(lam):
    '''
    mse loss that takes into account difference of corresponding arc lengths
    :param lam: regularization parameter for the arc difference
    '''

    def loss(output, target):
        sy = __summary_arc_length_sq(target)
        sy_hat = __summary_arc_length_sq(output)
        mse = torch.mean((output - target)**2)
        arc = torch.mean(torch.abs(sy_hat - sy))
        #cross = torch.mean((__normals(output) - __normals(target))**2)
        #der = torch.mean((__derivatives(output) - __derivatives(target))**2)

        #loss = mse + lam * arc + lam * cross + lam * der
        loss = mse + lam * arc
        return loss

    return loss


def __summary_arc_length_sq(tensor: torch.Tensor):
    diffs = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    sqdifs = torch.sum(diffs ** 2, dim=-1)
    return sqdifs

def __normals(tensor: torch.Tensor):
    diffs = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    diffs2 = tensor[:, :, 2:, :] - tensor[:, :, :-2, :]
    cross = torch.cross(diffs[:, :, :-1, :], diffs2)
    return cross


def __derivatives(tensor: torch.Tensor):
    diffs = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    return diffs
