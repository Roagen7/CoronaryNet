import torch


def manifold_likelihood_loss(lam_arc, lam_tor, lam_coh, lam_cvt):
    '''
    mse loss that takes into the shape of corresponding artery branches
    :param lam_arc: regularization parameter for the arc length difference
    :param lam_tor: regularization parameter for the torsion difference
    :param lam_coh: regularization parameter for the cohesion difference
    :param lam_cvt: regularization parameter for the curvature difference
    '''

    def loss(output, target):
            sy = __summary_arc_length_sq(target)
            sy_hat = __summary_arc_length_sq(output)
            ty = __compute_torsion(target)
            ty_hat = __compute_torsion(output)
            cy = target[:, :, 1:, :] - target[:, :, :-1, :]
            cy_hat = output[:, :, 1:, :] - output[:, :, :-1, :]
            cvy = compute_curvature(target)
            cvy_hat = compute_curvature(output)

            mse = torch.mean((output - target)**2)
            arc = torch.sum(torch.abs(sy_hat - sy))
            tor = torch.sum(torch.abs(ty_hat - ty))
            coh = torch.sum(torch.abs(cy_hat - cy))
            cvt = torch.sum(torch.abs(cvy_hat - cvy))

            loss = mse + lam_arc * arc + lam_tor * tor + lam_coh * coh + lam_cvt * cvt
            return loss

    return loss


def arc_loss(lam):
    '''
    mse loss that takes into account difference of corresponding arc lengths
    :param lam: regularization parameter for the arc difference
    '''

    def loss(output, target):
        sy = __summary_arc_length_sq(target)
        sy_hat = __summary_arc_length_sq(output)
        mse = torch.mean((output - target)**2)
        arc = torch.sum(torch.abs(sy_hat - sy))
        loss = mse + lam * arc
        return loss

    return loss


def __summary_arc_length_sq(tensor: torch.Tensor):
    diffs = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    sqdifs = torch.sum(diffs ** 2, dim=-1)
    return sqdifs


def __compute_torsion(tensor):
    vectors = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    cross_products = torch.cross(vectors[:, :, :-1, :], vectors[:, :, 1:, :], dim=-1)
    dot_products = torch.sum(vectors[:, :, :-1, :] * vectors[:, :, 1:, :], dim=-1)
    torsion = torch.atan2(torch.norm(cross_products, dim=-1), dot_products)
    sum_torsion = torch.sum(torsion, dim=-1)
    return sum_torsion


def compute_curvature(tensor):
    diff = tensor[:, :, 1:, :] - tensor[:, :, :-1, :]
    
    segment_lengths = torch.norm(diff, dim=3)
    
    diff_segment = diff[:, :, 1:, :] - diff[:, :, :-1, :]
    
    curvature = torch.norm(diff_segment, dim=3) / (segment_lengths[:, :, :-1] * segment_lengths[:, :, 1:])
    return curvature.mean(dim=2)


