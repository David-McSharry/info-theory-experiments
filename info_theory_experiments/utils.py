import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import scipy.io
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt, decimate
from scipy.fftpack import fft, fftfreq
import requests
import zipfile
import io


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def estimate_MI_smile(scores):
    """
    Returns the MI estimate using the SMILE estimator given the scores matrix and a clip
    """
    clip = 5

    first_term = scores.diag().mean()

    batch_size = scores.size(0)

    # clip scores between -clip and clip
    clipped_scores = torch.clamp(scores, -clip, clip)
    clipped_scores = scores

    # e^clipped_scores
    exp_clipped_scores = torch.exp(clipped_scores)

    mask = (torch.ones_like(exp_clipped_scores).to(device) - torch.eye(batch_size).to(device))

    masked_exp_clipped_scores = exp_clipped_scores * mask

    num_non_diag = mask.sum()

    mean_exp_clipped_scores = masked_exp_clipped_scores.sum() / num_non_diag

    second_term = torch.log2(mean_exp_clipped_scores)

    return (1/torch.log(torch.tensor(2.0))) * first_term - second_term


# https://neurotycho.brain.riken.jp/download/2012/20100802S1_Epidural-ECoG+Food-Tracking_B_Kentaro+Shimoda_mat_ECoG64-Motion6.zip



def prepare_batch(X):
    # take all samples except the last one as inputs
    input_data = X[:-1]

    # take all samples except the first one as targets
    target_data = X[1:]

    # stack them as pairs with dimension (999, 2, 10)
    pairs = torch.stack((input_data, target_data), dim=1)

    # assert pairs[0,0,0] == input_data[0,0]

    return pairs



def prepare_batch_and_randomize(X):
    """
    This function is for randomizing the input and targert time series steps 
    for control analysis
    """
    # take all samples except the last one as inputs
    input_data = X[:-1]

    # take all samples except the first one as targets
    target_data = X[1:]

    # Shuffle target and input data with different permutations
    input_indices = torch.randperm(input_data.size(0))
    target_indices = torch.randperm(target_data.size(0))
    input_data = input_data[input_indices]
    target_data = target_data[target_indices]



    # stack them as pairs with dimension (999, 2, 10)
    pairs = torch.stack((input_data, target_data), dim=1)

    # assert pairs[0,0,0] == input_data[0,0]

    return pairs





def create_glider_no_loop(grid, x, y):
    gliders = [
        torch.tensor([[0, 1, 0], [0, 0, 1], [1, 1, 1]], dtype=torch.float32),
        torch.tensor([[1, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=torch.float32),
        torch.tensor([[0, 0, 1], [1, 0, 1], [0, 1, 1]], dtype=torch.float32),
        torch.tensor([[1, 0, 0], [0, 1, 1], [1, 1, 0]], dtype=torch.float32)
    ]
    glider = gliders[torch.randint(0, len(gliders), (1,)).item()]
    grid[x:x+3, y:y+3] = glider

def update_grid_no_loop(grid):
    N = grid.shape[0]
    padded_grid = torch.zeros((N+2, N+2), dtype=torch.float32)
    padded_grid[1:-1, 1:-1] = grid
    
    total = (
        padded_grid[:-2, 1:-1] + padded_grid[2:, 1:-1] +
        padded_grid[1:-1, :-2] + padded_grid[1:-1, 2:] +
        padded_grid[:-2, :-2] + padded_grid[:-2, 2:] +
        padded_grid[2:, :-2] + padded_grid[2:, 2:]
    )
    
    new_grid = ((grid == 1) & ((total == 2) | (total == 3))) | ((grid == 0) & (total == 3))
    return new_grid.float()

def run_game_of_life_no_loop(N, max_steps, seed):
    torch.manual_seed(seed)
    
    grid = torch.zeros((N, N), dtype=torch.float32)
    x, y = torch.randint(0, N-3, (2,))
    create_glider_no_loop(grid, x, y)
    
    grids = [grid.clone()]
    for _ in range(max_steps - 1):
        new_grid = update_grid_no_loop(grid)
        if torch.all(new_grid == grid) or torch.sum(new_grid[0, :] + new_grid[-1, :] + new_grid[:, 0] + new_grid[:, -1]) > 0:
            break
        grid = new_grid
        grids.append(grid.clone())
    
    return torch.stack(grids)

def animate_game_of_life_no_loop(grids):
    fig, ax = plt.subplots()
    img = ax.imshow(grids[0].cpu().numpy(), interpolation='nearest', cmap='binary')
    
    def update(frame):
        img.set_array(grids[frame].cpu().numpy())
        return [img]
    
    ani = FuncAnimation(fig, update, frames=len(grids), interval=200, blit=True)
    plt.close(fig)  # Prevent the static plot from displaying
    return ani
