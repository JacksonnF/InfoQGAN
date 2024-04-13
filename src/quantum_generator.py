import pennylane as qml
from pennylane import numpy as np

number_of_qubits = 5
number_of_reps = 5
dev = qml.device("default.qubit", wires=number_of_qubits)


@qml.qnode(dev)
def quantum_generator(inputs, params):
    """
    Defines the quantum generator circuit. Embedding and Ansatz layers were based on the paper we followed.

    Args:
    - inputs (torch.Tensor): The input data.
    - params (torch.Tensor): The parameters of the circuit.

    Returns:
    - tuple: The probabilities of the qubits.
    """
    qml.AngleEmbedding(np.pi * inputs / 2, rotation="Y", wires=range(number_of_qubits))
    num_params = 0
    for layer in range(number_of_reps):
        # initial RX,Ry,RZ gates.
        for i in range(number_of_qubits):
            qml.RX(params[num_params], wires=i)
            qml.RY(params[num_params + 1], wires=i)
            qml.RZ(params[num_params + 2], wires=i)
            num_params += 3

        # Entangling block
        for i in range(number_of_qubits):
            qml.CNOT(wires=[i, (i + 1) % number_of_qubits])
    return qml.probs(wires=0), qml.probs(wires=1)


number_of_qubits_financial = 4
number_of_reps_financial = 5

dev_fin = qml.device("default.qubit", wires=number_of_qubits_financial)


@qml.qnode(dev_fin)
def quantum_generator_financial(inputs, params):
    """
    Defines the quantum generator circuit. Embedding and Ansatz layers were based on the paper we followed.

    Args:
    - inputs (torch.Tensor): The input data.
    - params (torch.Tensor): The parameters of the circuit.

    Returns:
    - tuple: The probabilities of the qubits.
    """
    qml.AngleEmbedding(
        np.pi * (inputs - 0.5) / 2, rotation="Y", wires=range(number_of_qubits)
    )
    num_params = 0
    for layer in range(number_of_reps_financial):
        # initial Ry gates.
        for i in range(number_of_qubits_financial):
            qml.RY(params[num_params], wires=i)
            num_params += 1

        # Entangling block
        for i in range(number_of_qubits_financial):
            qml.CZ(wires=[i, (i + 1) % number_of_qubits_financial])
    return qml.probs(wires=range(number_of_qubits_financial))
