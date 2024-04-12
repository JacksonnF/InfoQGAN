import torch

import numpy as np
import matplotlib.pyplot as plt


def plot_xy(x_pts, y_pts):
    plt.figure(figsize=(4, 4))
    plt.scatter(x_pts, y_pts, alpha=1, s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Test Generator Output")
    plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
    plt.legend()
    plt.show()
    return plt.figure()

def plot_xy_codes(x_pts, y_pts, code_index, rand_inp):
    plt.figure(figsize=(4, 4))
    plt.scatter(x_pts, y_pts, c=rand_inp[:, code_index], cmap='coolwarm', alpha=1, s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Feature Seperation for Code {code_index}")
    plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
    plt.legend()
    plt.colorbar(label='Code Value')
    plt.show()


class XYDistribution(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        return torch.stack((self.x[idx], self.y[idx]))
    
def boundary_adherence(points, center=(0.5, 0.5), side_length=0.5):
    x_min, y_min = center[0] - side_length / 2, center[1] - side_length / 2
    x_max, y_max = center[0] + side_length / 2, center[1] + side_length / 2

    within_bounds = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                     (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
    adherence_rate = np.mean(within_bounds) * 100  # Convert to percentage
    return adherence_rate
