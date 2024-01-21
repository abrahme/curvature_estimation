import torch
from typing import List
import torch.nn as nn
from .hsgp import HSGPExpQuadWithDerivative
from .manifold import GPRiemannianEuclidean


class RiemannianAutoencoder(nn.Module):

    def __init__(self, n: int,t: int,  m: List[int], c: List):
        super().__init__()

        d = int(n*(n-1)/2)
        self.gp_components = [HSGPExpQuadWithDerivative(m[d], c[d]) for _ in range(d)]
        self.metric_space = GPRiemannianEuclidean(n, self.gp_components)
        self.n = n
        self.d = d
        self.t = t

    def forward(self,initial_conditions):

        predicted_vals_geodesics = self.metric_space.metric.geodesic(initial_point = initial_conditions[:,0:self.n],
                                                                            initial_tangent_vec=initial_conditions[:,self.n:2*self.n])
        time_steps = torch.arange(0,1,self.t)
        predicted_vals = predicted_vals_geodesics(time_steps)
        return predicted_vals
    
    def loss(self, predicted_vals, actual_vals)


