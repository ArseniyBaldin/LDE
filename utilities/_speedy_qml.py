import math
import time

import numpy as np
import torch
import torch.nn as nn


# Timer
def timer(f, *args, return_out=False, **kwargs):
    """Times a function"""
    t = time.time()
    out = f(*args, **kwargs)
    t = time.time() - t
    if return_out:
        return t, out
    return t


# ENCODOING DATA
def tensor_product(matrices):
    """Performs Kronecker product on list of matrices"""
    M = torch.tensor(1)
    for matrix in matrices:
        M = torch.kron(M, matrix)
    return M


def binary_matrix(n_qubits, device=None, dtype=torch.float):
    """Creates a n by 2 ** n matrix where every column counts upwards in binary"""
    matrix = torch.zeros(n_qubits, 2**n_qubits, dtype=dtype)
    for i in range(n_qubits):
        ith_matrix = torch.zeros([2] * n_qubits)
        ith_slice = [slice(None)] * n_qubits
        ith_slice[i] = 1
        ith_matrix[tuple(ith_slice)] = 1
        ith_matrix = ith_matrix.reshape(-1)
        matrix[i] = ith_matrix

    if device is not None:
        matrix = matrix.to(device)
    return matrix


def rz_eigenvals(X, n_qubits=None, mask=None, ones=True):
    """Uses a binary matrix as the mask to create the eigenvalues of an RZ layer.
    Also, chunks the feature size of the data to fit into the number of qubits."""
    if X.dtype != torch.float:
        X = X.to(torch.float)

    n_features = X.shape[-1]

    if n_qubits is None:
        n_qubits = n_features

    if mask is None:
        mask = binary_matrix(n_qubits)

    if not ones:
        mask = mask * -2 + 1
    mask = mask.to(X.device)

    data_layers = -(-n_features // n_qubits)  # performs ceiling division

    rz_evs = []
    for x_chunk in X.chunk(data_layers, dim=-1):
        pad_size = n_qubits - x_chunk.shape[-1]
        x_chunk_padded = torch.nn.functional.pad(x_chunk, (0, pad_size))
        thetas = x_chunk_padded @ mask
        rz_evs.append(torch.exp(-1j * thetas / 2))

    return torch.stack(rz_evs, dim=0)


# CNOT
def cnot_perms(n_qubits, wires):
    """Creates permutation equivalent to general CNOT gate for n_qubits >= 2"""
    x = np.arange(2**n_qubits).reshape([2] * n_qubits)

    slice_a = [slice(None)] * n_qubits
    slice_b = [slice(None)] * n_qubits

    slice_a[wires[0]], slice_a[wires[1]] = 1, 1
    slice_b[wires[0]], slice_b[wires[1]] = 1, 0

    a, b = x[tuple(slice_a)].copy(), x[tuple(slice_b)].copy()
    x[tuple(slice_b)], x[tuple(slice_a)] = a, b
    return x.reshape(-1)


def cnot_ring(n_qubits):
    """Creates permutation equivalent to CNOT ring for n_qubits >= 2"""
    n2 = np.array([0, 2, 3, 1])  # returns precomputed result for 2 qubits
    if n_qubits == 2:
        return n2

    evens = np.zeros(2 ** (n_qubits - 1))  # half of future answers for even indices
    answer = n2[::2]
    for i in range(1, n_qubits - 1):
        evens[: 2**i] = answer
        a, b = np.array_split(answer, 2)
        evens[2**i : 2**i + 2 ** (i - 1)] = a + 3 * (2**i)
        evens[2**i + 2 ** (i - 1) : 2 ** (i + 1)] = b + (2**i)
        answer = evens[: 2 ** (i + 1)]

    a, b = np.split(
        evens.copy() + 2 ** (n_qubits - 2), 2
    )  # start to prepare the odd indices
    a = np.mod(a, 2 ** (n_qubits - 1))
    b, c = np.split(b, 2)
    b = b - 2 ** (n_qubits - 1)

    odds = np.concatenate((a, b, c), axis=0)[::-1]  # results for odd indices
    output = np.zeros(2**n_qubits)
    output[::2] = evens
    output[1::2] = odds
    return output.astype(int)


def cnot_incomplete_ring(n_qubits):
    """Creates permutation equivalent to CNOT ring for n_qubits >= 2"""
    n2 = np.array([0, 1, 3, 2])  # returns precomputed result for 2 qubits
    if n_qubits == 2:
        return n2

    evens = np.zeros(2 ** (n_qubits - 1))  # half of future answers for even indices
    answer = n2[::2]
    for i in range(1, n_qubits - 1):
        evens[: 2**i] = answer
        a, b = np.array_split(answer, 2)
        evens[2**i : 2**i + 2 ** (i - 1)] = a + 3 * (2**i)
        evens[2**i + 2 ** (i - 1) : 2 ** (i + 1)] = b + (2**i)
        answer = evens[: 2 ** (i + 1)]

    a, b = np.split(
        evens.copy() + 2 ** (n_qubits - 2), 2
    )  # start to prepare the odd indices
    a = np.mod(a, 2 ** (n_qubits - 1))
    b, c = np.split(b, 2)
    b = b - 2 ** (n_qubits - 1)

    odds = np.concatenate((a, b, c), axis=0)[::-1]
    start, end = np.split(odds, 2)
    odds = np.concatenate(
        (end, start), axis=0
    )  # last CNOT repeated to cancel previous one
    output = np.zeros(2**n_qubits)
    output[::2] = evens
    output[1::2] = odds
    return output.astype(int)


def cnot_staircase(up, down, n_qubits):
    """Creates CNOT staircase starting on up qubit and ending on down
    qubit (qubits are enumerated from 0)"""
    qubit_c = down - up + 1  # number of qubits in CNOT sequence
    answer = cnot_incomplete_ring(qubit_c)  # get the permutation

    for _ in range(up):  # add extra rows for qubits before CNOTs
        answer = np.concatenate((answer, answer + 2 ** (qubit_c)), axis=0)
        qubit_c += 1

    for _ in range(down + 1, n_qubits):  # shift rows for qubits after CNOTs
        evens = answer.copy() * 2  # every even is the original * 2
        odds = answer.copy() * 2 + 1  # every odd is the original * 2 + 1
        output = np.zeros(2**n_qubits)
        output[::2] = evens
        output[1::2] = odds
        answer = output

    return answer.astype(int)


def paulix_perms(n_qubits, wires):
    """Creates permutation equivalent to general pauli X gates applied to each wire in wires"""
    p = np.arange(2**n_qubits).reshape([2] * n_qubits)

    slice_a = [slice(None)] * n_qubits
    slice_b = [slice(None)] * n_qubits

    for i in wires:
        slice_a[i], slice_b[i] = 0, 1
        p[tuple(slice_a)], p[tuple(slice_b)] = (
            p[tuple(slice_b)],
            p[tuple(slice_a)].copy(),
        )
        slice_a[i], slice_b[i] = slice(None), slice(None)

    return p.reshape(-1).astype(int)


# hadamards
def had_small(phi, H=None):
    "Fastest hadamard layer for less than 8 qubits"
    if H is None:
        n_qubits = int(math.log2(phi.shape[-1]))
        h = torch.tensor([[1, 1], [1, -1]]).cfloat() * (2 ** (-1 / 2))
        H = tensor_product([h] * n_qubits)
    elif H.dtype != torch.cfloat:
        H = H.cfloat()
    H = H.to(phi.device)
    return phi @ H


def had_medium(phi, H=None):
    "Fastest hadamard layer for 8 to 11 qubits"
    if H is None:
        n_qubits = int(math.log2(phi.shape[-1]))
        h = torch.tensor([[1, 1], [1, -1]]).float() * (2 ** (-1 / 2))
        H = tensor_product([h] * n_qubits)
    elif H.dtype != torch.float:
        H = H.float()
    H = H.to(phi.device)
    return (phi.real @ H) + 1j * (phi.imag @ H)


def had_big(phi):
    "Fastest hadamard layer for 12 qubits or more"
    batch_size, state_size = phi.shape
    n_qubits = int(math.log2(state_size))
    phi = phi.clone().view(
        [batch_size] + [2] * n_qubits
    )  # Clone to avoid modifying the input tensor

    slices_a = [slice(None)] * (n_qubits + 1)
    slices_b = [slice(None)] * (n_qubits + 1)

    for i in range(1, n_qubits + 1):
        slices_a[i], slices_b[i] = 0, 1
        temp_a, temp_b = phi[tuple(slices_a)], phi[tuple(slices_b)]
        phi[tuple(slices_a)], phi[tuple(slices_b)] = temp_a + temp_b, temp_a - temp_b
        slices_a[i], slices_b[i] = slice(None), slice(None)
    return phi.reshape(batch_size, -1) * (2 ** -(n_qubits / 2))


def had(phi, H=None):
    "General hadamard layers"
    n_qubits = int(math.log2(phi.shape[-1]))
    if n_qubits <= 7:
        return had_small(phi, H=None)
    elif 7 < n_qubits <= 11:
        return had_medium(phi, H=None)
    else:
        return had_big(phi)


def sqrtZ(phi, conj=False):
    n_qubits = int(math.log2(phi.shape[-1]))
    S = tensor_product([torch.tensor([1, 1j])] * n_qubits).to(phi.device)
    if conj:
        S = S.conj()
    return phi * S


# Classes
class CNOT(nn.Module):
    """
    This class represents Speedy CNOT gates chain

    Args
    ____
    n_qubits : int
        Number of qubits
    """

    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.permutation = cnot_ring(n_qubits)

    def forward(self, phi):
        assert int(math.log2(phi.shape[-1])) == self.n_qubits
        return phi[:, self.permutation]


class Hadamard(nn.Module):
    """
    This class represents Speedy Hadamard gates

    Args
    ____
    n_qubits : int
        Number of qubits
    """

    def __init__(self, n_qubits=None):
        super().__init__()
        self.n_qubits = n_qubits

        self.H = None
        if n_qubits <= 11:
            h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * (2 ** (-1 / 2))
            self.H = tensor_product([h] * n_qubits)
            if n_qubits <= 7:
                self.H = self.H.cfloat()

    def forward(self, phi):
        if self.n_qubits is not None:
            assert int(math.log2(phi.shape[-1])) == self.n_qubits
        return had(phi, H=self.H)


class RZ(nn.Module):
    """
    This class represents RZ gates

    Args
    ____
    n_qubits : int
        Number of qubits
    """

    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.mask = binary_matrix(n_qubits)

    def forward(self, phi, thetas):
        assert int(math.log2(phi.shape[-1])) == self.n_qubits
        return phi * rz_eigenvals(thetas, mask=self.mask)[0]


class RX(nn.Module):
    """
    This class represents Speedy RX gates

    Args
    ____
    n_qubits : int
        Number of qubits
    """

    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.mask = binary_matrix(n_qubits)

        self.H = None
        if n_qubits <= 11:
            h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * (2 ** (-1 / 2))
            self.H = tensor_product([h] * n_qubits)
            if n_qubits <= 7:
                self.H = self.H.cfloat()

    def forward(self, phi, thetas):
        assert int(math.log2(phi.shape[-1])) == self.n_qubits
        phi = had(phi, H=self.H)
        phi = phi * rz_eigenvals(thetas, mask=self.mask)[0]
        phi = had(phi, H=self.H)
        return phi


class RY(nn.Module):
    """
    This class represents Speedy RY gates

    Args
    ____
    n_qubits : int
        Number of qubits
    """

    def __init__(self, n_qubits):
        super().__init__()
        self.n_qubits = n_qubits
        self.mask = binary_matrix(n_qubits)
        self.sqrtZ = tensor_product([torch.tensor([1, 1j])] * n_qubits)

        self.H = None
        if n_qubits <= 11:
            h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * (2 ** (-1 / 2))
            self.H = tensor_product([h] * n_qubits)
            if n_qubits <= 7:
                self.H = self.H.cfloat()

    def forward(self, phi, thetas):
        assert int(math.log2(phi.shape[-1])) == self.n_qubits
        phi = phi * self.sqrtZ.to(phi.device)
        phi = had(phi, H=self.H)
        phi = phi * rz_eigenvals(thetas, mask=self.mask)[0]
        phi = had(phi, H=self.H)
        phi = phi * self.sqrtZ.conj().to(phi.device)
        return phi


class Measurement(nn.Module):
    """
    This class represents Speedy measurements

    Args
    ____
    n_qubits : int
        Number of qubits

    observable : str
        Type of possible observations from ``["Z_0", "Z_all", "X_0", "X_all", "Y_0", "Y_all"]``
    """

    def __init__(self, n_qubits, observable="Z_all"):
        super().__init__()
        self.n_qubits = n_qubits

        Z_all = lambda phi: phi * tensor_product([torch.tensor([1, -1])] * n_qubits).to(
            phi.device
        )
        Z_0 = lambda phi: phi * tensor_product(
            [torch.tensor([1, -1])] + [torch.tensor([1, 1])] * (n_qubits - 1)
        ).to(phi.device)
        X_all = lambda phi: phi[:, torch.flip(torch.arange(2**n_qubits), dims=(0,))]
        X_0 = lambda phi: phi[
            :, torch.cat(torch.tensor_split(torch.arange(2**n_qubits), 2)[::-1])
        ]
        Y_all = lambda phi: X_all(Z_all(phi)) * (1j**n_qubits)
        Y_0 = lambda phi: X_0(Z_0(phi))

        self.observables = dict(
            zip(
                ["Z_0", "Z_all", "X_0", "X_all", "Y_0", "Y_all"],
                [Z_0, Z_all, X_0, X_all, Y_0, Y_all],
            )
        )
        self.observable = self.observables[observable]

    def forward(self, phi):
        return torch.einsum("...i,...i->...", self.observable(phi), phi.conj()).real


# Composite Models
class StrongEntanglerLayer(nn.Module):
    """
    This class represents Strong Entangler Layer
    """

    def __init__(self, n_qubits, RY=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.RY = RY
        self.cnot = cnot_ring(n_qubits)
        self.mask = binary_matrix(n_qubits)
        self.sqrtZ = tensor_product([torch.tensor([1, 1j])] * n_qubits)

        self.H = None
        if n_qubits <= 11:
            h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * (2 ** (-1 / 2))
            self.H = tensor_product([h] * n_qubits)
            if n_qubits <= 7:
                self.H = self.H.cfloat()

    def forward(self, phi, angles):
        assert int(math.log2(phi.shape[-1])) == self.n_qubits
        eigenvals = rz_eigenvals(angles, mask=self.mask)[0].cfloat()

        # these make the RX layer into RY
        if self.RY:
            eigenvals[0] = eigenvals[0] * self.sqrtZ.to(phi.device)
            eigenvals[2] = eigenvals[2] * self.sqrtZ.conj().to(phi.device)

        phi = phi * eigenvals[0]
        phi = had(phi, H=self.H)
        phi = phi * eigenvals[1]
        phi = had(phi, H=self.H)
        phi = phi * eigenvals[2]
        phi = phi[:, self.cnot]
        return phi


# State Vector Implementation
class StateVector(nn.Module):
    """
    This class represents State Vector implementation
    """

    def __init__(self, n_qubits, batchsize=1):
        super().__init__()
        self.batchsize = batchsize
        self.n_qubits = n_qubits

        self.state = self.data = None
        self.set_state()

        self.shape = self.state.shape
        self.cnot_p = cnot_ring(n_qubits)
        self.sqrtZ = tensor_product([torch.tensor([1, 1j])] * n_qubits)
        self.bin_mat = binary_matrix(n_qubits)

    def set_state(self, state=None):
        if state is None:
            state = torch.zeros(self.batchsize, 2**self.n_qubits, dtype=torch.cfloat)
            state[:, 0] = 1
        elif state == "random":
            state = torch.randn(self.batchsize, 2**self.n_qubits, dtype=torch.cfloat)
            state = torch.nn.functional.normalize(state, p=2, dim=-1)
        self.state = state

    def hadamard(self):
        self.state = had(self.state)

    def CNOT(self):
        self.state = self.state[:, self.cnot_p]

    def RZ(self, thetas):
        if len(thetas.shape) == 2:
            assert thetas.shape[0] == 1 or thetas.shape[0] == self.batchsize
        self.state = self.state * rz_eigenvals(thetas, mask=self.bin_mat)[0]

    def RX(self, thetas):
        self.hadamard()
        self.RZ(thetas)
        self.hadamard()

    def RY(self, thetas):
        self.state = self.state * self.sqrtZ
        self.RX(thetas)
        self.state = self.state * self.sqrtZ.conj()

    def measure(self, observable=None):
        Z_all = lambda phi: phi * tensor_product(
            [torch.tensor([1, -1])] * self.n_qubits
        )
        Z_0 = lambda phi: phi * tensor_product(
            [torch.tensor([1, -1])] + [torch.tensor([1, 1])] * (self.n_qubits - 1)
        )
        X_all = lambda phi: phi[
            :, torch.flip(torch.arange(2**self.n_qubits), dims=(0,))
        ]
        X_0 = lambda phi: phi[
            :, torch.cat(torch.tensor_split(torch.arange(2**self.n_qubits), 2)[::-1])
        ]
        Y_all = lambda phi: X_all(Z_all(phi)) * (1j**self.n_qubits)
        Y_0 = lambda phi: X_0(Z_0(phi))

        observables = dict(
            zip(
                ["Z_0", "Z_all", "X_0", "X_all", "Y_0", "Y_all"],
                [Z_0, Z_all, X_0, X_all, Y_0, Y_all],
            )
        )

        if observable is not None:
            obs = observables[observable]
            return torch.einsum(
                "...i,...i->...", obs(self.state), self.state.conj()
            ).real
        else:
            return torch.abs(self.state) ** 2

    def __repr__(self):
        return f"StateVector({self.state})"


# Testing with cuda
def RX2_func(phi, thetas):
    "CUDA faster function"
    batch_size, state_size = phi.shape
    n_qubits = int(math.log2(state_size))
    phi = phi.clone().view(
        [batch_size] + [2] * n_qubits
    )  # Clone to avoid modifying the input tensor

    slices_a = [slice(None)] * (n_qubits + 1)
    slices_b = [slice(None)] * (n_qubits + 1)

    c = torch.cos(thetas / 2).view([batch_size, n_qubits] + [1] * (n_qubits - 1))
    s = -1j * torch.sin(thetas / 2).view([batch_size, n_qubits] + [1] * (n_qubits - 1))

    for i in range(1, n_qubits + 1):
        slices_a[i], slices_b[i] = 0, 1
        temp_a, temp_b = phi[tuple(slices_a)], phi[tuple(slices_b)]
        phi[tuple(slices_a)], phi[tuple(slices_b)] = (
            c[:, i - 1] * temp_a + s[:, i - 1] * temp_b,
            s[:, i - 1] * temp_a + c[:, i - 1] * temp_b,
        )
        slices_a[i], slices_b[i] = slice(None), slice(None)
    return phi.reshape(batch_size, -1)


class RX2(nn.Module):
    """
    This class represents RX2 gate
    """

    def __init__(self, n_qubits=None):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, phi, thetas):
        if self.n_qubits is not None:
            assert int(math.log2(phi.shape[-1])) == self.n_qubits
        return RX2_func(phi, thetas)


def RY2_func(phi, thetas):
    "CUDA faster function"
    batch_size, state_size = phi.shape
    n_qubits = int(math.log2(state_size))
    phi = phi.clone().view(
        [batch_size] + [2] * n_qubits
    )  # Clone to avoid modifying the input tensor

    slices_a = [slice(None)] * (n_qubits + 1)
    slices_b = [slice(None)] * (n_qubits + 1)

    c = torch.cos(thetas / 2).view([batch_size, n_qubits] + [1] * (n_qubits - 1))
    s = torch.sin(thetas / 2).view([batch_size, n_qubits] + [1] * (n_qubits - 1))

    for i in range(1, n_qubits + 1):
        slices_a[i], slices_b[i] = 0, 1
        temp_a, temp_b = phi[tuple(slices_a)], phi[tuple(slices_b)]
        phi[tuple(slices_a)], phi[tuple(slices_b)] = (
            c[:, i - 1] * temp_a + s[:, i - 1] * temp_b,
            -s[:, i - 1] * temp_a + c[:, i - 1] * temp_b,
        )
        slices_a[i], slices_b[i] = slice(None), slice(None)
    return phi.reshape(batch_size, -1)


class RY2(nn.Module):
    """
    This class represents RY2 gate
    """

    def __init__(self, n_qubits=None):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, phi, thetas):
        if self.n_qubits is not None:
            assert int(math.log2(phi.shape[-1])) == self.n_qubits
        return RY2_func(phi, thetas)


def RZ2_func(phi, thetas):
    "CUDA faster function"

    batch_size, state_size = phi.shape
    n_qubits = int(math.log2(state_size))
    phi = phi.clone().view(
        [batch_size] + [2] * n_qubits
    )  # Clone to avoid modifying the input tensor

    slices_a = [slice(None)] * (n_qubits + 1)
    slices_b = [slice(None)] * (n_qubits + 1)

    e1 = torch.exp(-1j * thetas / 2).view([batch_size, n_qubits] + [1] * (n_qubits - 1))
    e2 = e1.conj()

    for i in range(1, n_qubits + 1):
        slices_a[i], slices_b[i] = 0, 1
        temp_a, temp_b = phi[tuple(slices_a)], phi[tuple(slices_b)]
        phi[tuple(slices_a)], phi[tuple(slices_b)] = (
            e1[:, i - 1] * temp_a,
            e2[:, i - 1] * temp_b,
        )
        slices_a[i], slices_b[i] = slice(None), slice(None)
    return phi.reshape(batch_size, -1)


class RZ2(nn.Module):
    """
    This class represents RZ2 gate
    """

    def __init__(self, n_qubits=None):
        super().__init__()
        self.n_qubits = n_qubits

    def forward(self, phi, thetas):
        if self.n_qubits is not None:
            assert int(math.log2(phi.shape[-1])) == self.n_qubits
        return RZ2_func(phi, thetas)


class StrongEntanglerLayer2(nn.Module):
    """
    This class represents Strong Entangler Layer 2
    """

    def __init__(self, n_qubits, RY=True):
        super().__init__()
        self.n_qubits = n_qubits
        self.RY = RY
        self.cnot = cnot_ring(n_qubits)

        self.H = None
        if n_qubits <= 11:
            h = torch.tensor([[1.0, 1.0], [1.0, -1.0]]) * (2 ** (-1 / 2))
            self.H = tensor_product([h] * n_qubits)
            if n_qubits <= 7:
                self.H = self.H.cfloat()

    def forward(self, phi, angles):
        assert int(math.log2(phi.shape[-1])) == self.n_qubits
        phi = RZ2_func(phi, angles[0])
        if self.RY:
            phi = RY2_func(phi, angles[1])
        else:
            phi = RX2_func(phi, angles[1])
        phi = RZ2_func(phi, angles[2])
        phi = phi[:, self.cnot]
        return phi
