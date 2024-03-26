import torch
import pandas as pd
import numpy as np
from models.train import train_irregular_timesteps
from models.utils import save_model
from geomstats.geometry.hypersphere import Hypersphere


data = pd.read_csv("whale_data/clean_data.csv")
keep_whales = np.random.choice(pd.unique(data["individual-local-identifier"]), 10)
data = data[data["individual-local-identifier"].isin(keep_whales)]
feature_data = pd.read_csv("whale_data/feature_data.csv")
feature_data = feature_data[feature_data["individual-local-identifier"].isin(keep_whales)]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trajectory_list =[ torch.from_numpy(item).to(device).float() for item in data.groupby("individual-local-identifier").apply(lambda x: x[["x","y","z"]].to_numpy()).to_list()]
timestep_list = [ torch.from_numpy(item).to(device).float() for item in data.groupby("individual-local-identifier").apply(lambda x: x["total_seconds_normalized"].to_numpy()).to_list()]

initial_velocity = np.concatenate(feature_data.groupby("individual-local-identifier").apply(lambda x: Hypersphere(2).to_tangent((x.head(1)[["x","y","z"]].to_numpy() - x.tail(1)[["x","y","z"]].to_numpy())/ x.tail(1)["total_seconds_normalized"].to_numpy(), x.head(1)[["x","y","z"]].to_numpy())).to_list())
initial_position = np.concatenate(feature_data.groupby("individual-local-identifier").apply(lambda x: x.head(1)[["x","y","z"]].to_numpy()).to_list())
initial_conditions = torch.from_numpy(np.hstack([initial_position, initial_velocity])).to(device).float()


model, predictions = train_irregular_timesteps(trajectory_list, initial_conditions, 1, 3, timestep_list, hidden_dim=100)


save_model(model,1000, "whale")

