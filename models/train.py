from typing import List, Tuple
import torch
import numpy as np
import torch.optim as optim
from .model import RiemannianAutoencoder, SymmetricRiemannianAutoencoder, VanillaAutoencoder, SymmetricRiemannianAutoencoderSphere


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



def train(input_trajectories, initial_conditions: torch.Tensor, val_input_trajectories, 
          val_initial_conditions:Tuple[torch.Tensor, torch.Tensor],  epochs, 
          regularizer:float, n, t, m:List[int], c:float, basis, active_dims: List, return_preds:bool = False, val: bool = False, loss_type:str = "L2"):
    
    model = RiemannianAutoencoder(n =n,t = t,m = m,c = c,regularization=regularizer, basis = basis, active_dims = active_dims, loss_type =loss_type)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    preds = []  
    with torch.no_grad():
        predicted_trajectories = model.forward(initial_conditions)
        preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories = model.forward(initial_conditions)
        loss = model.loss(torch.permute(predicted_trajectories, (1,0,2)), input_trajectories.float())
        # Backward pass and optimization
        loss.backward(retain_graph=True)
        
        optimizer.step()

        if val:
            model.eval()
            with torch.no_grad():
                predicted_val_trajectories = model.forward(val_initial_conditions)
                val_loss = model.loss(torch.permute(predicted_val_trajectories.float().detach(), (1,0,2)), val_input_trajectories.float(), val = True)
        else:
            val_loss = "None"
        if return_preds:
            preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")



    return model, preds




def train_symmetric_circle(input_trajectories, initial_conditions: torch.Tensor, val_input_trajectories, 
          val_initial_conditions:Tuple[torch.Tensor, torch.Tensor],  epochs, n, t, m:List[int], c:float, basis, active_dims: List, return_preds:bool = False, val: bool = False, loss_type:str = "L2"):
    
    model = SymmetricRiemannianAutoencoder(n = n,t = t,m = m,c = c, basis = basis, active_dims = active_dims, loss_type =loss_type)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    preds = []  
    with torch.no_grad():
        predicted_trajectories = model.forward(initial_conditions)
        preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories = model.forward(initial_conditions)
        loss = model.loss(torch.permute(predicted_trajectories, (1,0,2)), input_trajectories.float())
        # Backward pass and optimization
        loss.backward(retain_graph=True)
        
        optimizer.step()

        if val:
            model.eval()
            with torch.no_grad():
                predicted_val_trajectories = model.forward(val_initial_conditions)
                val_loss = model.loss(torch.permute(predicted_val_trajectories.float().detach(), (1,0,2)), val_input_trajectories.float(), val = True)
        else:
            val_loss = "None"
        if return_preds:
            preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")


    return model, preds




def train_symmetric_sphere(input_trajectories, initial_conditions: torch.Tensor, val_input_trajectories, 
          val_initial_conditions:Tuple[torch.Tensor, torch.Tensor],  epochs, n, t, m:List[int], c:float, basis, active_dims: List, return_preds:bool = False, val: bool = False, loss_type:str = "L2"):
    
    model = SymmetricRiemannianAutoencoderSphere(n = n,t = t,m = m,c = c, basis = basis, active_dims = active_dims, loss_type =loss_type)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    preds = []  
    with torch.no_grad():
        predicted_trajectories = model.forward(initial_conditions)
        preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories = model.forward(initial_conditions)
        loss = model.loss(torch.permute(predicted_trajectories, (1,0,2)), input_trajectories.float())
        # Backward pass and optimization
        loss.backward(retain_graph=True)
        optimizer.step()

        if val:
            model.eval()
            with torch.no_grad():
                predicted_val_trajectories = model.forward(val_initial_conditions)
                val_loss = model.loss(torch.permute(predicted_val_trajectories.float().detach(), (1,0,2)), val_input_trajectories.float(), val = True)
        else:
            val_loss = "None"
        if return_preds:
            preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")


    return model, preds


def train_vanilla_autoencoder(input_trajectories,  val_input_trajectories, epochs, timesteps: int, ndims: int, return_preds:bool = False, val: bool = False, hidden_size:int = 25):
    input_traj_reshape = input_trajectories.view( input_trajectories.size(0), timesteps*ndims)
    val_input_traj_reshape = val_input_trajectories.view(val_input_trajectories.size(0), timesteps*ndims)
    model = VanillaAutoencoder(input_size=input_traj_reshape.shape[1], hidden_size=hidden_size)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    preds = []  
    with torch.no_grad():
        predicted_trajectories = model.forward(input_traj_reshape)
        preds.append(torch.reshape(predicted_trajectories, (input_trajectories.size(0), timesteps, ndims)))
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories = model.forward(input_traj_reshape)
        loss = model.loss(input_traj_reshape)
        # Backward pass and optimization
        loss.backward()
        
        optimizer.step()

        if val:
            model.eval()
            with torch.no_grad():
                predicted_val_trajectories = model.forward(val_input_traj_reshape)
                val_loss = model.loss(val_input_traj_reshape)
        else:
            val_loss = "None"
        if return_preds:
            preds.append(torch.reshape(predicted_val_trajectories, (val_input_trajectories.size(0), timesteps, ndims)))

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss.item()}")



    return model, preds