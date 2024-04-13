import torch
from torch.utils.data import Dataset

import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import pandas as pd


def plot_xy(x_pts, y_pts):
    plt.figure(figsize=(4, 4))
    plt.scatter(x_pts, y_pts, alpha=1, s=2)
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
    plt.scatter(x_pts, y_pts, c=rand_inp[:, code_index], cmap='coolwarm', alpha=1, s=2)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"Feature Seperation for Code {code_index}")
    plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
    plt.legend()
    plt.colorbar(label='Code Value')
    plt.show()

def plot_xy_overlaid(x_pts, y_pts, x, y, label='Central Square'):
    plt.figure(figsize=(4, 4))
    plt.scatter(x, y, alpha=1, s=2, label='Target Distribution')
    plt.scatter(x_pts, y_pts, alpha=1, s=2, label='InfoQGAN')
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title(f"InfoQGAN {label}")
    plt.gca().set_aspect('equal', adjustable='box')  # Make the plot square
    plt.legend()
    plt.show()
    return plt.figure()

def graph_distribution_financial(distribution):
    """
    Graphs the distribution.

    Args:
    - distribution (torch.Tensor): The probability distribution tensor.
    """
    
    # Calculate bin centers
    bin_edges = np.linspace(-0.1, 0.1, 17)  # 16 bins means 17 edges
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Average of edges to find center

    # Create a bar plot
    plt.figure(figsize=(5, 3))
    plt.bar(bin_centers, distribution, width=bin_edges[1] - bin_edges[0], align='center', color='blue', alpha=0.7)
    plt.xlabel('Return Interval')
    plt.ylabel('Probability')
    plt.title('Probability Distribution of Returns')
    plt.show()


class XYDistribution(Dataset):
    def __init__(self, x, y):
        """

        Args:
            x (list): List of x values.
            y (list): List of y values.
        """
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        
    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.x)
    
    def __getitem__(self, idx):
        """
        Get an item from the dataset at the given index.

        Args:
            idx (int): Index of the item to retrieve.

        Returns:
            torch.Tensor: A tensor containing the x and y values at the given index.
        """
        return torch.stack((self.x[idx], self.y[idx]))
    
def boundary_adherence(points, center=(0.5, 0.5), side_length=0.5):
    """
    Calculates the adherence rate of points within a specified boundary.

    Parameters:
    - points (numpy.ndarray): Array of points with shape (n, 2), where n is the number of points.
    - center (tuple, optional): Center coordinates of the boundary. Default is (0.5, 0.5).
    - side_length (float, optional): Length of the side of the boundary. Default is 0.5.

    Returns:
    - adherence_rate (float): The percentage of points that fall within the boundary.
    """
    x_min, y_min = center[0] - side_length / 2, center[1] - side_length / 2
    x_max, y_max = center[0] + side_length / 2, center[1] + side_length / 2

    within_bounds = ((points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                     (points[:, 1] >= y_min) & (points[:, 1] <= y_max))
    adherence_rate = np.mean(within_bounds) * 100
    return adherence_rate

def generate_biased_circle(num_points):
    np.random.seed(0)
    theta = np.random.uniform(0, 2 * np.pi, num_points)
    r = np.random.beta(a=2, b=4, size=num_points)

    x = r * np.cos(theta)
    y = r * np.sin(theta)

    x_bias = 0.3  
    y_bias = 0.3  
    x = x * x_bias + 0.3
    y = y * y_bias + 0.3

    return x, y

def generate_central_square(num_points):
    np.random.seed(0)
    x = np.random.uniform(low=0.25, high=0.75, size=num_points)
    y = np.random.uniform(low=0.25, high=0.75, size=num_points)

    return x, y

def fetch_and_process_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    daily_returns = stock_data['Adj Close'].pct_change().dropna()
    normalized_returns = np.clip(daily_returns, -0.1, 0.1)
    return normalized_returns

def generate_stock_dataset():
    aapl_returns = fetch_and_process_data('AAPL', '2011-01-01', '2022-12-31')
    tsla_returns = fetch_and_process_data('TSLA', '2011-01-01', '2022-12-31')

    aligned_returns = pd.concat([aapl_returns, tsla_returns], axis=1, join='inner').dropna()

    aapl_returns, tsla_returns = aligned_returns.iloc[:, 0], aligned_returns.iloc[:, 1]

    num_datasets = 2000
    num_bins = 16
    datasets = []

    for _ in range(num_datasets):
        alpha = np.random.uniform(0, 1) 
        portfolio_returns = alpha * aapl_returns + (1 - alpha) * tsla_returns
        
        bin_edges = np.linspace(-0.1, 0.1, num_bins+1)
        hist, _ = np.histogram(portfolio_returns, bins=bin_edges)
        datasets.append(hist/len(aapl_returns))

    return np.array(datasets)

class DistributionDataset(Dataset):
    def __init__(self, datasets):
        """
        Initializes the dataset with a list of probability distributions.

        Args:
            distributions (list of list of float): List of probability distributions.
        """
        self.distributions = torch.tensor(datasets, dtype=torch.float32)  # Probability distributions as torch tensor
        
    def __len__(self):
        """
        Returns the total number of distributions in the dataset.
        
        Returns:
            int: The number of distributions in the dataset.
        """
        return len(self.distributions)
    
    def __getitem__(self, idx):
        """
        Retrieves a distribution by index.

        Args:
            idx (int): The index of the distribution to retrieve.

        Returns:
            torch.Tensor: The probability distribution tensor at the given index.
        """
        return self.distributions[idx]