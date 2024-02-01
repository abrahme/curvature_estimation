import torch
from geomloss.samples_loss import SamplesLoss
from torchdyn.core import NeuralODE
from typing import List
import torch.nn as nn
from .hsgp import HSGPExpQuadWithDerivative
from .manifold import GaussianProcessRiemmanianMetric, GaussianProcessRiemmanianMetricSymmetricCircle, GaussianProcessRiemmanianMetricSymmetricSphere



class _ODEFunc(nn.Module):
    def __init__(self, module: GaussianProcessRiemmanianMetric):
        super(_ODEFunc, self).__init__()
        self.module = module
    def forward(self, t, x, *args, **kwargs):
        ode_eq = self.module.geodesic_equation(torch.hsplit(x,2), t)
        return ode_eq

class ODEBlockMLP(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = 'dopri5',
                 sensitivity:str = 'adjoint',
                 rtol: float = 1e-4, atol: float = 1e-4):
        super(ODEBlockMLP, self).__init__()
        self.odefunc = odefunc
        self.ode_solver = NeuralODE(self.odefunc, solver = solver, 
                                    sensitivity=sensitivity, atol=atol, rtol=rtol)
    
    def forward(self, X: torch.Tensor, integration_time):

        _, out = self.ode_solver(X, integration_time)
        return out

class ODEBlock(nn.Module):
    def __init__(self, odefunc: nn.Module, solver: str = 'dopri5',
                 sensitivity:str = 'autograd',
                 rtol: float = 1e-4, atol: float = 1e-4):
        super(ODEBlock, self).__init__()
        self.odefunc = _ODEFunc(odefunc)
        self.ode_solver = NeuralODE(self.odefunc, solver = solver, 
                                    sensitivity=sensitivity, atol=atol, rtol=rtol)
    
    def forward(self, X: torch.Tensor, integration_time):

        _, out = self.ode_solver(X, integration_time)
        return out

class RiemannianAutoencoder(nn.Module):

    def __init__(self, n: int,t: int,  m: List[int], c: float, regularization: float, basis, active_dims: List, loss_type: str = "L2"):
        super(RiemannianAutoencoder, self).__init__()
        d = int(n*(n+1)/2)
        self.gp_components = nn.ModuleList([HSGPExpQuadWithDerivative(m, c, active_dims) for _ in range(d)])
        for gp_component in self.gp_components:
            gp_component.prior_linearized(basis) ### initialize basis
        self.metric_space = GaussianProcessRiemmanianMetric(n, self.gp_components)
        self.ode_layer = ODEBlock(self.metric_space)
        self.n = n ### dimension of manifold
        self.d = d ### number of free functions
        self.t = t ### timepoints to extend
        self.regularization = regularization
        self.basis = basis
        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[:,:,0:split_size]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        elif self.loss_type == "Hausdorff":
            reconstruction_loss = torch.FloatTensor([0.0])
            for i in range(predicted_vals.shape[0]):
                reconstruction_loss += SamplesLoss("sinkhorn")(predicted_vals[i], actual_vals[i])
        if not val:
            reconstruction_loss = reconstruction_loss + self.parameter_loss() * (1/(10 ** (self.n + 1)))
        
        if self.regularization > 0:
            if not val:
                reconstruction_loss = self.prior_loss() + reconstruction_loss
        return reconstruction_loss
    
    def parameter_loss(self):
        parameter_loss = torch.FloatTensor([0.0])
        for gp in self.gp_components:
            parameter_loss += torch.square(gp._beta).mean()/len(self.gp_components)
            parameter_loss += torch.mean(gp.ls)/len(self.gp_components)
        return parameter_loss

    
    def prior_loss(self):
        #### lie derivative loss with symmetry of circle
        ### TODO generalize to other symmetries 
        christoffels = self.metric_space.christoffels(self.basis)
        prior_loss = nn.MSELoss()(self.basis[:,0].view(-1,1,1)*christoffels[:,1,0:2,0:2], self.basis[:,1].view(-1,1,1)*christoffels[:,0,0:2,0:2])
        return prior_loss * torch.FloatTensor([self.regularization]) 



class RiemannianAutoencoderBatch(RiemannianAutoencoder):

    def __init__(self, n: int,t: List[torch.Tensor],  m: List[int], c: float, regularization: float, basis, active_dims: List, loss_type: str = "L2"):
        super(RiemannianAutoencoderBatch, self).__init__(n = n, t = t, m = m, c = c, regularization=regularization, basis=basis, active_dims=active_dims,loss_type=loss_type)


    def forward(self,initial_conditions, i: int):
        time_steps = self.t[i]
        predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[:,:,0:split_size]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        elif self.loss_type == "Hausdorff":
            reconstruction_loss = torch.FloatTensor([0.0])
            for i in range(predicted_vals.shape[0]):
                reconstruction_loss += SamplesLoss("sinkhorn")(predicted_vals[i], actual_vals[i])
        if not val:
            reconstruction_loss = reconstruction_loss + self.parameter_loss() * (1/(10 ** (self.n + 1)))
        
        if self.regularization > 0:
            if not val:
                reconstruction_loss = self.prior_loss() + reconstruction_loss
        return reconstruction_loss
    
    def parameter_loss(self):
        parameter_loss = torch.FloatTensor([0.0])
        for gp in self.gp_components:
            parameter_loss += torch.square(gp._beta).mean()/len(self.gp_components)
            parameter_loss += torch.mean(gp.ls)/len(self.gp_components)
        return parameter_loss

    
    def prior_loss(self):
        #### lie derivative loss with symmetry of circle
        ### TODO generalize to other symmetries 
        christoffels = self.metric_space.christoffels(self.basis)
        prior_loss = torch.FloatTensor([self.regularization]) * torch.square(self.basis[:,0]*(christoffels[:,1,:,:].sum((-1,-2))) - self.basis[:,1]*(christoffels[:,0,:,:].sum((-1,-2)))).mean()
        return prior_loss

class SymmetricRiemannianAutoencoder(nn.Module):
    def __init__(self, n: int,t: int,  m: List[int], c: float,  basis, active_dims: List, loss_type: str = "L2"):
        super(SymmetricRiemannianAutoencoder, self).__init__()
        d = int(n*(n+1)/2)
        self.gp_components = nn.ModuleList([HSGPExpQuadWithDerivative(m, c, active_dims) for _ in range(d)])
        for gp_component in self.gp_components:
            gp_component.prior_linearized(basis) ### initialize basis
        self.metric_space = GaussianProcessRiemmanianMetricSymmetricCircle(n, self.gp_components)
        self.ode_layer = ODEBlock(self.metric_space)
        self.n = n ### dimension of manifold
        self.d = d ### number of free functions
        self.t = t ### timepoints to extend

        self.basis = basis
        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[:,:,0:split_size]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        elif self.loss_type == "Hausdorff":
            reconstruction_loss = torch.FloatTensor([0.0])
            for i in range(predicted_vals.shape[0]):
                reconstruction_loss += SamplesLoss("sinkhorn")(predicted_vals[i], actual_vals[i])
        if not val:
            reconstruction_loss = reconstruction_loss + self.parameter_loss() * (1/(10 ** (self.n + 1)))
        return reconstruction_loss
    
    def parameter_loss(self):
        parameter_loss = torch.FloatTensor([0.0])
        for gp in self.gp_components:
            parameter_loss += torch.square(gp._beta).mean()/len(self.gp_components)
            parameter_loss += torch.mean(gp.ls)/len(self.gp_components)

        return parameter_loss


class SymmetricRiemannianAutoencoderSphere(nn.Module):
    def __init__(self, n: int,t: int,  m: List[int], c: float,  basis, active_dims: List, loss_type: str = "L2"):
        super(SymmetricRiemannianAutoencoderSphere, self).__init__()
        d = int(n*(n+1)/2)
        self.gp_components = nn.ModuleList([HSGPExpQuadWithDerivative(m, c, active_dims) for _ in range(d)])
        for gp_component in self.gp_components:
            gp_component.prior_linearized(basis) ### initialize basis
        self.metric_space = GaussianProcessRiemmanianMetricSymmetricSphere(n, self.gp_components)
        self.ode_layer = ODEBlock(self.metric_space)
        self.n = n ### dimension of manifold
        self.d = d ### number of free functions
        self.t = t ### timepoints to extend

        self.basis = basis
        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[:,:,0:split_size]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        elif self.loss_type == "Hausdorff":
            reconstruction_loss = torch.FloatTensor([0.0])
            for i in range(predicted_vals.shape[0]):
                reconstruction_loss += SamplesLoss("sinkhorn")(predicted_vals[i], actual_vals[i])
        if not val:
            reconstruction_loss = reconstruction_loss + self.parameter_loss() * (1/(10 ** (self.n + 1)))
        return reconstruction_loss
    
    def parameter_loss(self):
        parameter_loss = torch.FloatTensor([0.0])
        for gp in self.gp_components:
            parameter_loss += torch.square(gp._beta).mean()/len(self.gp_components)
            parameter_loss += torch.mean(gp.ls)/len(self.gp_components)

        return parameter_loss
    
class VanillaAutoencoder(nn.Module):
    def __init__(self, n: int,t: int, loss_type: str = "L2"):
        super(VanillaAutoencoder, self).__init__()

        self.ode_layer = ODEBlockMLP(nn.Sequential(
        nn.Linear(2*n, 16),
        nn.Tanh(),
        nn.Linear(16, 2*n)
    ))
        self.n = n ### dimension of manifold
        self.t = t ### timepoints to extend

        self.loss_type = loss_type

    def forward(self,initial_conditions):
        time_steps = torch.linspace(0.0,1.0,self.t)
        predicted_vals = self.ode_layer(initial_conditions, time_steps)
        split_size = self.n
        return predicted_vals[:,:,0:split_size]

    
    def loss(self, predicted_vals, actual_vals, val = False):
        if self.loss_type == "L2":
            reconstruction_loss = nn.MSELoss()(predicted_vals,actual_vals)
        elif self.loss_type == "Hausdorff":
            reconstruction_loss = torch.FloatTensor([0.0])
            for i in range(predicted_vals.shape[0]):
                reconstruction_loss += SamplesLoss("sinkhorn")(predicted_vals[i], actual_vals[i])
        return reconstruction_loss
    












