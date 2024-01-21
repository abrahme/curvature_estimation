import torch
from typing import List
import torch.nn as nn
from .hsgp import HSGPExpQuadWithDerivative
from .manifold import GPRiemannianEuclidean


class RiemannianAutoencoder(nn.Module):

    def __init__(self, n: int,t: int,  m: List[int], c: List, regularization: float, basis):
        super().__init__()

        d = int(n*(n-1)/2)
        self.gp_components = [HSGPExpQuadWithDerivative(m[d], c[d]) for _ in range(d)]
        for gp_component in self.gp_components:
            gp_component.prior_linearized(basis) ### initialize basis
        self.metric_space = GPRiemannianEuclidean(n, self.gp_components)
        self.n = n ### dimension of manifold
        self.d = d ### number of free functions
        self.t = t ### timepoints to extend
        self.regularization = regularization
        self.basis = basis

    def forward(self,initial_conditions):

        predicted_vals_geodesics = self.metric_space.metric.geodesic(initial_point = initial_conditions[:,0:self.n],
                                                                            initial_tangent_vec=initial_conditions[:,self.n:2*self.n])
        time_steps = torch.arange(0,1,self.t)
        predicted_vals = predicted_vals_geodesics(time_steps)
        return predicted_vals
    
    def loss(self, predicted_vals, actual_vals):

        reconstruction_loss =  nn.MSELoss(reduce=True, reduction="sum")(predicted_vals, actual_vals) 
        parameter_loss = 0
        for gp in self.gp_components:
            parameter_loss += nn.MSELoss(reduce=True, reduction="sum")(gp._beta, 0)
            parameter_loss += torch.sum(gp.ls)
        
        return reconstruction_loss + parameter_loss + self.regularization*self.prior_loss()
    
    def prior_loss(self):
        #### lie derivative loss with symmetry of circle
        ### TODO generalize to other symmetries 
        christoffels = self.metric_space.metric.christoffels(self.basis)
        return nn.MSELoss(reduce=True, reduction="sum")(self.basis[:,0]*(christoffels[:,1,:,:].sum(axis=(-1,-2))), self.basis[:,1]*(christoffels[:,0,:,:].sum(axis=(-1,-2))))




