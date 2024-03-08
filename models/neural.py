import torch 
import torch.nn as nn
import numpy as np

class PSD(nn.Module):
    '''A Neural Net which outputs a positive semi-definite matrix'''
    def __init__(self, input_dim, hidden_dim, diag_dim):
        super(PSD, self).__init__()
        self.diag_dim = diag_dim
        self.off_diag_dim = int(diag_dim * (diag_dim - 1) / 2)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, self.diag_dim + self.off_diag_dim)

        for l in [self.linear1, self.linear2]:
            nn.init.orthogonal_(l.weight) # use a principled initialization
        
        self.nonlinearity = nn.Tanh()

    def forward(self, q):

        bs = q.shape[0]
        h = self.nonlinearity( self.linear1(q) )
        diag, off_diag = torch.split(self.linear2(h), [self.diag_dim, self.off_diag_dim], dim=1)

        L = torch.diag_embed(nn.Softplus()(diag))

        ind = np.tril_indices(self.diag_dim, k=-1)
        flat_ind = np.ravel_multi_index(ind, (self.diag_dim, self.diag_dim))
        L = torch.flatten(L, start_dim=1)
        L[:, flat_ind] = off_diag
        L = torch.reshape(L, (bs, self.diag_dim, self.diag_dim))

        D = torch.bmm(L, L.permute(0, 2, 1))
        return D