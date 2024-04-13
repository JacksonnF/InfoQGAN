import torch
import torch.nn as nn
import pennylane as qml
from pennylane import numpy as np

from .quantum_generator import (
    quantum_generator,
    quantum_generator_financial,
    number_of_qubits,
    number_of_qubits_financial,
    number_of_reps,
    number_of_reps_financial,
)


class Discriminator(nn.Module):
    def __init__(self, input_size, hid_size, use_financial=False) -> None:
        """
        Args:
        - input_size (int): The size of the input data.
        - hid_size (int): The size of the hidden layers.
        """
        super().__init__()
        self.use_financial = use_financial
        self.layers = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Performs a forward pass through the Discriminator.

        Args:
        - x (torch.Tensor): The input data.

        Returns:
        - torch.Tensor: The output of the Discriminator.
        """
        return self.layers(x)

    def fit_discriminator(self, x_data, criterion, net_G, optimizer, input_size):
        """
        Trains the Discriminator module.

        Args:
        - x_data (torch.Tensor): The real data.
        - criterion: The loss criterion.
        - net_G: The Generator network.
        - optimizer: The optimizer.
        - input_size (int): The size of the input data.

        Returns:
        - float: The loss value.

        """
        batch_size, _ = x_data.shape[0], x_data.shape[1]
        if self.use_financial:
            x_rand = torch.rand((batch_size, input_size))
        else:
            x_rand = 2 * torch.rand((batch_size, input_size)) - 1

        self.zero_grad()

        # Forward pass Discriminator on "real" data
        labels_real = torch.ones((batch_size, 1)) * 0.9  # Label smoothing
        outputs = self.forward(x_data)
        loss_d_real = criterion(outputs, labels_real)

        # Forward pass Discriminator with "fake" data from Generator
        g = net_G(x_rand).detach()  # Stop gradients from being updated in generator
        labels_fk = torch.zeros((batch_size, 1)) + 0.1  # Label smoothing
        outputs = self.forward(g)
        loss_d_fake = criterion(outputs, labels_fk)

        loss_d = loss_d_fake + loss_d_real
        loss_d.backward()  # Compute Gradients
        optimizer.step()  # Update Weights
        return loss_d.item()


class Generator(nn.Module):
    def __init__(self, use_financial=False) -> None:
        super().__init__()
        self.use_financial = use_financial
        if self.use_financial:
            weights_financial = {
                "params": number_of_qubits_financial * number_of_reps_financial
            }
            self.q_gen = qml.qnn.TorchLayer(
                quantum_generator_financial, weights_financial
            )
        else:
            weights = {"params": 3 * number_of_qubits * number_of_reps}
            self.q_gen = qml.qnn.TorchLayer(quantum_generator, weights)

    def forward(self, x):
        """
        Performs a forward pass through the Generator.

        Args:
        - x (torch.Tensor): The input data.

        Returns:
        - torch.Tensor: The output of the Generator.
        """
        if self.use_financial:
            return self.q_gen(x)
        else:
            out = (4 / np.pi) * torch.arcsin(torch.sqrt(self.q_gen(x))) - 1 / 2
            return out[:, [0, 2]]

    def fit_generator(
        self, net_D, T, batch_size, input_size, criterion, optimizer, code_dim, beta
    ):
        """
        Trains the Generator module.

        Args:
        - net_D: The Discriminator network.
        - T: The T network.
        - batch_size (int): The size of the batch.
        - input_size (int): The size of the input data.
        - criterion: The loss criterion.
        - optimizer: The optimizer.
        - code_dim (int): The dimension of the code.
        - beta (float): The beta value.

        Returns:
        - float: The loss value.
        """
        if self.use_financial:
            x_rand = torch.rand((batch_size, input_size))
        else:
            x_rand = 2 * torch.rand((batch_size, input_size)) - 1
        self.zero_grad()

        # Generate outputs With Generator and check if they fool Discriminator
        labels_real = torch.ones((batch_size, 1)) * 0.9
        g = self.forward(x_rand)
        outputs = net_D(g)

        loss_g = criterion(
            outputs, labels_real
        )  # We want "fake" Generator output to look real

        # Compute the MINE loss term: -(E_pxy[T(x,y)] - log(E_pxpy[e^T(x,y)]))
        # Note: term 1 uses joint pdf, term 2 uses marginal pdfs
        x_marg = x_rand[torch.randperm(x_rand.size(0)), 0:code_dim]
        mine = torch.mean(T(g, x_rand[:, 0:code_dim])) - torch.log(
            torch.mean(torch.exp(T(g, x_marg)))
        )
        loss_g += -mine * beta

        loss_g.backward()
        optimizer.step()

        return loss_g.item()


class T(nn.Module):
    def __init__(self, gen_out_dim, code_dim, hid_size, use_financial=False) -> None:
        """
        Initializes the T module.

        Args:
        - gen_out_dim (int): The dimension of the generator output.
        - code_dim (int): The dimension of the code.
        - hid_size (int): The size of the hidden layers.
        """
        super().__init__()
        self.use_financial = use_financial
        self.layers = nn.Sequential(
            nn.Linear(code_dim + gen_out_dim, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, hid_size),
            nn.ReLU(),
            nn.Linear(hid_size, 1),
        )

    def forward(self, gen, codes):
        """
        Performs a forward pass through the T module.

        Args:
        - gen (torch.Tensor): The Generator output.
        - codes (torch.Tensor): The code data.

        Returns:
        - torch.Tensor: The output of the T module.

        """
        return self.layers(torch.cat((gen, codes), dim=1))

    def fit_t(self, net_G, batch_size, input_size, code_dim, optimizer):
        """
        Trains the T module.

        Args:
        - net_G: The Generator network.
        - batch_size (int): The size of the batch.
        - input_size (int): The size of the input data.
        - code_dim (int): The dimension of the code.
        - optimizer: The optimizer.

        Returns:
        - float: The loss value.
        """
        if self.use_financial:
            x_rand = torch.rand((batch_size, input_size))
        else:
            x_rand = 2 * torch.rand((batch_size, input_size)) - 1
        self.zero_grad()

        # Compute loss following similar logic to MINE term in generator loss
        with torch.no_grad():
            g = net_G(x_rand)

        T_out = self.forward(g, x_rand[:, 0:code_dim])
        x_marg = x_rand[torch.randperm(x_rand.size(0)), 0:code_dim]

        t_loss = -(
            torch.mean(T_out)
            - torch.log(torch.mean(torch.exp(self.forward(g, x_marg))))
        )
        t_loss.backward()
        optimizer.step()

        return t_loss.item()
