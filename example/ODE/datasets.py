import torch
from random import random
from torch.utils.data import Dataset

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

#################################################
# sphere dataset for different dimensions
#################################################
def random_point_in_sphere(dim, min_radius, max_radius):
    """Returns a point sampled uniformly at random from a sphere if min_radius
    is 0. Else samples a point approximately uniformly on annulus.
    Parameters
    ----------
    dim : int
        Dimension of sphere
    min_radius : float
        Minimum distance of sampled point from origin.
    max_radius : float
        Maximum distance of sampled point from origin.
    """
    # Sample distance of point from origin
    unif = random()
    distance = (max_radius - min_radius) * (unif ** (1. / dim)) + min_radius
    # Sample direction of point away from origin
    direction = torch.randn(dim)
    unit_direction = direction / torch.norm(direction, 2)
    return distance * unit_direction


class ConcentricSphere(Dataset):
    """Dataset of concentric d-dimensional spheres. Points in the inner sphere
    are mapped to -1, while points in the outer sphere are mapped 1.
    Parameters
    ----------
    dim : int
        Dimension of spheres.
    inner_range : (float, float)
        Minimum and maximum radius of inner sphere. For example if inner_range
        is (1., 2.) then all points in inner sphere will lie a distance of
        between 1.0 and 2.0 from the origin.
    outer_range : (float, float)
        Minimum and maximum radius of outer sphere.
    num_points_inner : int
        Number of points in inner cluster
    num_points_outer : int
        Number of points in outer cluster
    """
    def __init__(self, dim, inner_range, outer_range, num_points_inner,
                 num_points_outer):
        self.dim = dim
        self.inner_range = inner_range
        self.outer_range = outer_range
        self.num_points_inner = num_points_inner
        self.num_points_outer = num_points_outer

        self.data = []
        self.targets = []

        # Generate data for inner sphere
        for _ in range(self.num_points_inner):
            self.data.append(
                random_point_in_sphere(dim, inner_range[0], inner_range[1])
            )
            self.targets.append(torch.LongTensor([0]))

        # Generate data for outer sphere
        for _ in range(self.num_points_outer):
            self.data.append(
                random_point_in_sphere(dim, outer_range[0], outer_range[1])
            )
            self.targets.append(torch.LongTensor([1]))

    def __getitem__(self, index):
        return self.data[index], self.targets[index]

    def __len__(self):
        return len(self.data)