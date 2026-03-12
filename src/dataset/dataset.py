import math
import torch
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, P, N, D, d, sigma, seed=None, on_sphere=True, coefficients="binary", L=None):
        """
        P: Number of patterns
        N: Number of sites
        D: Number of features in the random field
        d: Dimensionality of each site
        sigma: Standard deviation of the Gaussian noise
        on_sphere: If True, normalize data to lie on a sphere
        coefficients: Type of coefficients ('binary' or 'gaussian')
        """
        self.P = P
        self.N = N
        self.D = D
        self.d = d
        self.sigma = sigma
        self.on_sphere = on_sphere
        self.coefficients = coefficients
        if L == None:
            self.L = D
        else:
            self.L = L

        if seed is not None:
            torch.manual_seed(seed)

        # xi of shape [P, N, d]
        self.xi = torch.randn(P, N, d) * sigma

        if self.D > 0:
            self.RF()

        # Normalize xi along last dimension if on_sphere is True
        if self.on_sphere:
            self.xi = self.normalize(self.xi)

    def RF(self, seed=None):
        if seed is not None:
            torch.manual_seed(seed)
        self.f = torch.randint(0, 2, (self.D, self.N, self.d)).float() * 2 - 1
        # Normalize f along the last dimension if on_sphere is True
        if self.on_sphere:
            self.f = self.normalize(self.f)

        # Create tensor c of shape [P, D] 
        if self.coefficients == "binary":
            self.c = torch.randint(0, 2, (self.P, self.D)).float() * 2 - 1  # Random +1 or -1
        elif self.coefficients == "gaussian":
            self.c = torch.randn(self.P, self.D)  # Gaussian random numbers
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")
        self.c = self.c / math.sqrt(self.D)

        #For each row of self.c, set to zero self.D-L random entries among the self.D. For each row, different random entries to set to zero
        indices_to_zero = torch.rand(self.P, self.D)
        indices_to_zero = indices_to_zero.argsort(dim=1)[:,:self.D-self.L]
        self.c = self.c.scatter(1, indices_to_zero, 0)

        # Update xi
        self.xi = torch.einsum('pk,kia->pia', self.c, self.f)

        mask = self.xi == 0

        random_binary_values = torch.randint(0, 2, self.xi.shape, device=self.xi.device) * 2 - 1
        
        self.xi = torch.where(mask, random_binary_values.float(), self.xi.float())
        if self.on_sphere:
            self.xi = self.normalize(self.xi)

    def get_generalization(self, P_hat):  #P_hat number of generalization vectors to get, L number of features to use
        # Create tensor c of shape [P_hat, N] 
        if self.coefficients == "binary":
            c = torch.randint(0, 2, (P_hat, self.D)).float() * 2 - 1  # Random +1 or -1
        elif self.coefficients == "gaussian":
            c = torch.randn(P_hat, self.D)  # Gaussian random numbers
        else:
            raise ValueError("coefficients must be 'binary' or 'gaussian'")

        c = c / math.sqrt(self.D)

        #For each row of self.c, set to zero self.D-L random entries among the self.D. For each row, different random entries to set to zero
        indices_to_zero = torch.rand(P_hat, self.D)
        indices_to_zero = indices_to_zero.argsort(dim=1)[:,:self.D-self.L]
        c = c.scatter(1, indices_to_zero, 0)

        self.xi_new = torch.einsum('pk,kia->pia', c, self.f)

        # Create a mask where self.xi == 0
        mask = self.xi_new == 0

        random_binary_values = torch.randint(0, 2, self.xi_new.shape, device=self.xi_new.device) * 2 - 1
        
        self.xi = torch.where(mask, random_binary_values.float(), self.xi.float())
        if self.on_sphere:
            self.xi_new = self.normalize(self.xi_new)
        return self.xi_new


    def normalize(self, x):
        # Normalize each d-dimensional vector in x along the last dimension
        norms = x.norm(dim=-1, keepdim=True)+1e-9
        return x / norms

    def __len__(self):
        return self.P

    def __getitem__(self, index):
        return self.xi[index]

class DatasetF(Dataset):
    def __init__(self, D, f):
        self.D = D
        self.f = f

    def __len__(self):
        return self.D

    def __getitem__(self, index):
        return self.f[index]

