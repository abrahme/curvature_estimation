from typing import List, Tuple
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from .model import RiemannianAutoencoder, VanillaAutoencoder, GPRiemannianAutoencoder


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
        self.val_loss_min = 0
        self.delta = delta
        self.trace_func = trace_func
    def __call__(self, val_loss):

        score = val_loss

        if self.best_score is None:
            self.best_score = score
        elif score > self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0



def train(input_trajectories, initial_conditions: torch.Tensor, val_input_trajectories, 
          val_initial_conditions:Tuple[torch.Tensor, torch.Tensor],  epochs,  n, t, hidden_dim = 20,  return_preds:bool = False, val: bool = False, loss_type:str = "L2", model_type:str = "neural"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiemannianAutoencoder(n =n,t = t, loss_type =loss_type, hidden_dim=hidden_dim) if model_type == "neural" else GPRiemannianAutoencoder(n = n, t = t, loss_type=loss_type, basis = initial_conditions[...,:n].to(device), m_val=hidden_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0) if model_type != "neural" else optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    preds = []  
    early_stopping = EarlyStopping(verbose = True, patience=90, delta = 0.0)
    with torch.no_grad():
        predicted_trajectories = model.forward(initial_conditions.to(device))
        preds.append(torch.permute(predicted_trajectories.to(device).detach(), (1,0,2)))
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        predicted_trajectories = model.forward(initial_conditions.to(device))
        loss = nn.MSELoss()(torch.permute(predicted_trajectories.to(device), (1,0,2)), input_trajectories.to(device).float())
        # Backward pass and optimization
        loss.backward()
        
        optimizer.step()

        if val:
            model.eval()
            with torch.no_grad():
                predicted_val_trajectories = model.forward(val_initial_conditions.to(device))
                val_loss = nn.MSELoss()(torch.permute(predicted_val_trajectories.float().detach(), (1,0,2)), val_input_trajectories.to(device).float())
        else:
            val_loss = "None"
        if return_preds:
            preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))
        
        early_stopping(val_loss=val_loss if val else 0.0)

        if early_stopping.early_stop:
            return model, preds, val_loss

        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")




    return model, preds, val_loss

def train_irregular_timesteps(input_trajectories: List[torch.tensor], initial_conditions: torch.Tensor, epochs,  n, t: List[torch.tensor], 
                              hidden_dim = 20, loss_type:str = "L2", model_type:str = "neural"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RiemannianAutoencoder(n =n,t = 1, loss_type =loss_type, hidden_dim=hidden_dim) if model_type == "neural" else GPRiemannianAutoencoder(n = n, t = 1, loss_type=loss_type, basis = initial_conditions[...,:n].to(device), m_val=hidden_dim)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0) if model_type != "neural" else optim.Adam(model.parameters(), lr=0.01, weight_decay=0.0)
    preds = []  
    for epoch in range(epochs):
        optimizer.zero_grad()
        # Forward pass
        loss = torch.tensor([0.0]).to(device)
        for i, input_trajectory in enumerate(input_trajectories):
            predicted_trajectory = model.forward(initial_conditions[i][None, ...], t[i])
            loss += nn.MSELoss()(torch.permute(predicted_trajectory, (1,0,2)), input_trajectory[None,...])

            if epoch == epochs - 1:
                preds.append(predicted_trajectory)
        # Backward pass and optimization
        loss.backward()
        
        optimizer.step()



        print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: None")




    return model, [pred.detach() for pred in preds]



# def train_group(input_trajectories, initial_conditions: torch.Tensor, val_input_trajectories, 
#           val_initial_conditions:Tuple[torch.Tensor, torch.Tensor],  epochs,  n, t, rep_in, rep_out, group,  hidden_dim = 20,  return_preds:bool = False, val: bool = False, loss_type:str = "L2"):
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = GroupRiemannianAutoencoder(n = n, t = t, hidden_dim=hidden_dim, rep_in = rep_in, rep_out=rep_out, group = group, loss_type=loss_type)
#     model.to(device)
#     optimizer =  optim.Adam(model.parameters(), lr=0.01)
#     preds = []  
#     with torch.no_grad():
#         predicted_trajectories = model.forward(initial_conditions.to(device))
#         preds.append(torch.permute(predicted_trajectories.to(device).detach(), (1,0,2)))
#     for epoch in range(epochs):
#         optimizer.zero_grad()
#         # Forward pass
#         predicted_trajectories = model.forward(initial_conditions.to(device))
#         loss = nn.MSELoss()(torch.permute(predicted_trajectories.to(device), (1,0,2)), input_trajectories.to(device).float())
#         # Backward pass and optimization
#         loss.backward()
        
#         optimizer.step()

#         if val:
#             model.eval()
#             with torch.no_grad():
#                 predicted_val_trajectories = model.forward(val_initial_conditions.to(device))
#                 val_loss = nn.MSELoss()(torch.permute(predicted_val_trajectories.float().detach(), (1,0,2)), val_input_trajectories.to(device).float())
#         else:
#             val_loss = "None"
#         if return_preds:
#             preds.append(torch.permute(predicted_trajectories.detach(), (1,0,2)))

#         print(f"Epoch: {epoch + 1}, Loss: {loss.item()}, Val Loss: {val_loss}")



#     return model, preds



def train_vanilla_autoencoder(input_trajectories, initial_conditions: torch.Tensor, val_input_trajectories, 
          val_initial_conditions:Tuple[torch.Tensor, torch.Tensor], epochs, 
           n, t, return_preds:bool = False, val: bool = False, loss_type:str = "L2"):

    
    model = VanillaAutoencoder(n = n, t = t, loss_type= loss_type)
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
        loss.backward()
        
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