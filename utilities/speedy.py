import math
from functools import cached_property

import numpy as np
import torch
from torch import nn

from _speedy_qml import (
    cnot_perms,
    had_big,
    had_medium,
    had_small,
    paulix_perms,
    rz_eigenvals,
    tensor_product,
)
# from tqml.tqnet.exceptions import DimensionException


#############################################
#    SpeedyLayer (base class)               #
#############################################
class SpeedyLayer(nn.Module):
    """
    Base class for speedy QML layers.
    Contains shared functions and cached properties that can be reused.
    """

    def __init__(
            self,
            in_features,
            n_qubits,
            depth,
            measurement_mode="None",
            rotation="Z",
            entangling="strong",
            measure="Y",
            hw_efficient=False,
            device=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.n_qubits = n_qubits
        self.depth = depth
        self.measurement_mode = measurement_mode
        self.rotation = rotation
        self.entangling = entangling
        self.measure = measure
        self.hw_efficient = hw_efficient
        self.device = device

        if self.measurement_mode == "single":
            self.out_features = 1
        elif self.measurement_mode == "even":
            self.out_features = math.ceil(self.n_qubits / 2)
        elif self.measurement_mode == "None":
            self.out_features = self.n_qubits

    ##############################################################
    #   Amplitude Embedding method (NEW!)                       #
    ##############################################################
    @staticmethod
    def amplitude_embedding(x: torch.Tensor) -> torch.Tensor:
        """
        Normalizes the input vector(s) and returns them as amplitudes.

        Args:
            x (torch.Tensor): Shape (..., 2^n_qubits).
                              Each row (or last dimension) is an unnormalized state vector.

        Returns:
            torch.Tensor: Shape (..., 2^n_qubits), normalized across the last dimension.
        """
        # Compute norms along the last dimension
        # Keep dimension so that broadcasting works for division.
        norm = torch.norm(x, dim=-1, keepdim=True)
        # Avoid division by zero if the input happens to be all-zeros
        norm = torch.where(norm == 0.0, torch.ones_like(norm), norm)
        return x / norm

    #########################
    #   Cached Properties   #
    #########################
    @cached_property
    def H(self):
        h = torch.tensor([[1, 1], [1, -1]]) * (2 ** (-1 / 2))
        h_layer = tensor_product([h] * self.n_qubits)
        return h_layer

    @cached_property
    def sqrtZ(self):
        # sqrtZ = diag(1, i) for 1 qubit
        S = tensor_product([torch.tensor([1, 1j])] * self.n_qubits)
        return S

    @cached_property
    def binary_matrix(self, ones=True):
        """Creates an n_qubits x (2**n_qubits) matrix where each column is a binary count."""
        matrix = torch.zeros(self.n_qubits, 2 ** self.n_qubits)
        for i in range(self.n_qubits):
            ith_matrix = torch.zeros([2] * self.n_qubits)
            ith_slice = [slice(None)] * self.n_qubits
            ith_slice[i] = 1
            ith_matrix[tuple(ith_slice)] = 1
            ith_matrix = ith_matrix.reshape(-1)
            matrix[i] = ith_matrix

        if ones:
            matrix = matrix * -2 + 1
        return matrix

    @cached_property
    def cnot_ring(self):
        """Creates permutation equivalent to CNOT ring for self.n_qubits >= 2"""
        n2 = np.array([0, 2, 3, 1])
        if self.n_qubits == 2:
            return n2

        evens = np.zeros(2 ** (self.n_qubits - 1))
        answer = n2[::2]
        for i in range(1, self.n_qubits - 1):
            evens[: 2 ** i] = answer
            a, b = np.array_split(answer, 2)
            evens[2 ** i: 2 ** i + 2 ** (i - 1)] = a + 3 * (2 ** i)
            evens[2 ** i + 2 ** (i - 1): 2 ** (i + 1)] = b + (2 ** i)
            answer = evens[: 2 ** (i + 1)]

        a, b = np.split(evens.copy() + 2 ** (self.n_qubits - 2), 2)
        a = np.mod(a, 2 ** (self.n_qubits - 1))
        b, c = np.split(b, 2)
        b = b - 2 ** (self.n_qubits - 1)
        odds = np.concatenate((a, b, c), axis=0)[::-1]

        output = np.zeros(2 ** self.n_qubits)
        output[::2] = evens
        output[1::2] = odds
        return output.astype(int)

    @cached_property
    def cnot_incomplete_ring(self):
        """Creates permutation equivalent to CNOT ring for self.n_qubits >= 2"""
        n2 = np.array([0, 1, 3, 2])
        if self.n_qubits == 2:
            return n2

        evens = np.zeros(2 ** (self.n_qubits - 1))
        answer = n2[::2]
        for i in range(1, self.n_qubits - 1):
            evens[: 2 ** i] = answer
            a, b = np.array_split(answer, 2)
            evens[2 ** i: 2 ** i + 2 ** (i - 1)] = a + 3 * (2 ** i)
            evens[2 ** i + 2 ** (i - 1): 2 ** (i + 1)] = b + (2 ** i)
            answer = evens[: 2 ** (i + 1)]

        a, b = np.split(evens.copy() + 2 ** (self.n_qubits - 2), 2)
        a = np.mod(a, 2 ** (self.n_qubits - 1))
        b, c = np.split(b, 2)
        b = b - 2 ** (self.n_qubits - 1)

        odds = np.concatenate((a, b, c), axis=0)[::-1]
        start, end = np.split(odds, 2)
        odds = np.concatenate((end, start), axis=0)
        output = np.zeros(2 ** self.n_qubits)
        output[::2] = evens
        output[1::2] = odds
        return output.astype(int)

    @cached_property
    def cnot_even_meas(self):
        p_0 = np.arange(0, 2 ** self.n_qubits)
        for i in range(0, self.n_qubits, 2):
            if i + 1 in range(self.n_qubits):
                p = cnot_perms(self.n_qubits, wires=(i + 1, i))
                p_0 = p_0[p]
        return p_0

    @cached_property
    def init_state(self):
        state = torch.zeros(1, 2 ** self.n_qubits, dtype=torch.cfloat)
        state[:, 0] = 1
        return state


    ############################################
    #               Measurement               #
    ############################################
    def meas(self, phi):
        """
        Measurement in X, Y, Z, or computational basis probabilities.
        Also includes the option to return the quantum state itself ("state").
        """
        if self.measurement_mode == "probs":
            probabilities = torch.abs(phi) ** 2
            probabilities = probabilities / probabilities.sum(dim=-1, keepdim=True)
            self.out_features = 2 ** self.n_qubits
            return probabilities

        # Add a new measurement mode for returning the state vector
        if self.measurement_mode == "state":
            self.out_features = phi.shape[-1]  # Number of amplitudes in the state
            return phi

        # Build the Pauli operators / perms
        if self.measure == "Z":
            if self.measurement_mode == "single":
                # single qubit Z
                obs = lambda phi: phi * tensor_product(
                    [torch.tensor([1, -1])] * self.n_qubits
                ).to(phi.device)
            elif self.measurement_mode == "even":
                obs = lambda phi: torch.stack(
                    [
                        phi
                        * tensor_product(
                            [torch.tensor([1, 1])] * n
                            + [torch.tensor([1, -1])]
                            + [torch.tensor([1, 1])] * (self.n_qubits - 1 - n)
                        )
                        for n in range(0, self.n_qubits, 2)
                    ],
                    dim=-1,
                ).to(phi.device)
            else:
                obs = lambda phi: torch.stack(
                    [
                        phi
                        * tensor_product(
                            [torch.tensor([1, 1])] * n
                            + [torch.tensor([1, -1])]
                            + [torch.tensor([1, 1])] * (self.n_qubits - 1 - n)
                        )
                        for n in range(self.n_qubits)
                    ],
                    dim=-1,
                ).to(phi.device)

        else:
            # measure in X or Y (via permutations)
            if self.measurement_mode == "single":
                obs = lambda phi: phi[:, np.arange(2 ** self.n_qubits - 1, -1, -1)]
            elif self.measurement_mode == "even":
                obs = lambda phi: torch.stack(
                    [
                        phi[:, paulix_perms(self.n_qubits, [n])]
                        for n in range(0, self.n_qubits, 2)
                    ],
                    dim=-1,
                ).to(phi.device)
            else:
                obs = lambda phi: torch.stack(
                    [
                        phi[:, paulix_perms(self.n_qubits, [n])]
                        for n in range(self.n_qubits)
                    ],
                    dim=-1,
                ).to(phi.device)

        # Apply measurement
        if self.measurement_mode == "single":
            temp_meas = lambda phi: torch.einsum(
                "...i,...i->...", obs(phi), phi.conj()
            ).real
            self.out_features = 1
        elif self.measurement_mode == "even":
            temp_meas = lambda phi: torch.einsum(
                "...ij,...i->...j", obs(phi), phi.conj()
            ).real
            self.out_features = math.ceil(self.n_qubits / 2)
        else:  # "None"
            temp_meas = lambda phi: torch.einsum(
                "...ij,...i->...j", obs(phi), phi.conj()
            ).real
            self.out_features = self.n_qubits

        if self.measurement_mode == "even":
            # reorder phi for even measurement
            phi = phi[:, self.cnot_even_meas]
        if self.measure == "Y":
            # multiply by sqrtZ^\dagger before measurement
            phi = phi * self.sqrtZ.conj().to(phi.device)

        return temp_meas(phi)


    ############################################
    #                 1-Qubit Gates            #
    ############################################
    def had(self, phi):
        """Applies global Hadamard to phi."""
        if self.n_qubits <= 7:
            return had_small(phi, H=self.H)
        elif 7 < self.n_qubits <= 11:
            return had_medium(phi, H=self.H)
        else:
            return had_big(phi)

    def rz(self, phi, eigenvals):
        return phi * eigenvals

    def rx(self, phi, eigenvals):
        phi = self.had(phi)
        phi = phi * eigenvals
        phi = self.had(phi)
        return phi

    def ry(self, phi, eigenvals):
        phi = phi * self.sqrtZ.conj().to(phi.device)
        phi = self.had(phi)
        phi = phi * eigenvals
        phi = self.had(phi)
        phi = phi * self.sqrtZ.to(phi.device)
        return phi

    ############################################
    #           2-Qubit Entangling Gates       #
    ############################################
    def bel_forward(self, phi, eigenvals_w):
        """
        Basic entangling layer:
          RX -> ring-of-CNOT
        """
        phi = self.rx(phi, eigenvals_w[0])
        if not self.hw_efficient:
            phi = phi[:, self.cnot_ring]
        else:
            phi = phi[:, self.cnot_incomplete_ring]
        return phi

    def sel_forward(self, phi, eigenvals_w):
        """
        Strong entangling layer:
          RZ -> RY -> RZ -> ring-of-CNOT
        """
        phi = self.rz(phi, eigenvals_w[0])
        phi = self.ry(phi, eigenvals_w[1])
        phi = self.rz(phi, eigenvals_w[2])
        if not self.hw_efficient:
            phi = phi[:, self.cnot_ring]
        else:
            phi = phi[:, self.cnot_incomplete_ring]
        return phi


#############################################
#       SpeedyVQ (Angle Embedding)          #
#############################################
class SpeedyVQ(SpeedyLayer):
    r"""
    A simple Quantum Layer using angle embedding for inputs.
    ...
    """

    def __init__(
            self,
            in_features,
            depth,
            measurement_mode="None",
            rotation="Z",
            entangling="strong",
            measure="Y",
            hw_efficient=False,
            device=None,
    ):
        super().__init__(
            in_features,
            in_features,  # Here, n_qubits = in_features (which is unusual for big n!)
            depth,
            measurement_mode=measurement_mode,
            rotation=rotation,
            entangling=entangling,
            measure=measure,
            hw_efficient=hw_efficient,
            device=device,
        )

        # picking rotation
        self.embed_rot = {"X": self.rx, "Y": self.ry, "Z": self.rz}[rotation]

        # picking entangler
        self.entangler_forward = {
            "basic": self.bel_forward,
            "strong": self.sel_forward,
        }[entangling]

        # initialize weights
        weights_per_layer = 3 if entangling == "strong" else 1
        self.weights = nn.Parameter(
            torch.rand(1, depth, weights_per_layer, self.n_qubits) * 2 * torch.pi
        )

    def forward(self, x):
        """
        1) Angle-embed x
        2) Repeated entangling
        3) Measure
        """
        batched = True
        if len(x.shape) == 1:
            x = x[None, :]
            batched = False
        if batched:
            batch_size = x.shape[0]

        # Convert real values of x into eigenvalues for RZ
        # (the so-called angle embedding)
        eigenvals_x = rz_eigenvals(x, mask=self.binary_matrix)[0].cfloat().to(x.device)
        eigenvals_w = (
            rz_eigenvals(self.weights, mask=self.binary_matrix)[0].cfloat().to(x.device)
        )

        # Start in |00..0> state
        phi = self.init_state.to(x.device)

        # Step 1) Embedding using rotation-based method
        phi = self.embed_rot(phi, eigenvals_x)

        # Step 2) Parametrized entangling layers
        for j in range(self.depth):
            phi = self.entangler_forward(phi, eigenvals_w[0, j])

        # Step 3) Measurement
        out = self.meas(phi)

        if batched:
            out = out.view(batch_size, -1)
        else:
            out = out.view(-1)
        return out


#############################################
#    VQAmplitudeEmbedding (NEW!)            #
#############################################
class VQAmplitudeEmbedding(SpeedyLayer):
    r"""
    A simple Quantum Layer using amplitude embedding for inputs
    instead of angle embedding. Otherwise, the structure is
    analogous to SpeedyVQ:
      - amplitude embedding
      - repeated entangling
      - measure
    """

    def __init__(
            self,
            in_features: int,
            depth: int,
            measurement_mode: str = "None",
            entangling: str = "strong",
            measure: str = "Y",
            hw_efficient: bool = False,
            device=None,
    ):
        """
        For amplitude embedding to work in a standard way, in_features
        should match 2^n_qubits. That is, if in_features=8, then n_qubits=3, etc.
        """
        # Make sure in_features = 2**n_qubits
        # e.g. if in_features=8 => n_qubits=3
        #     if in_features=4 => n_qubits=2
        # Check integer log2:
        n_qubits_float = math.log2(in_features)
        if abs(n_qubits_float - round(n_qubits_float)) > 1e-9:
            raise DimensionException(
                f"For amplitude embedding, in_features={in_features} must be 2^n_qubits!"
            )
        n_qubits = int(round(n_qubits_float))

        super().__init__(
            in_features=in_features,
            n_qubits=n_qubits,
            depth=depth,
            measurement_mode=measurement_mode,
            rotation="Z",  # rotation won't be used for amplitude, but we'll keep the param
            entangling=entangling,
            measure=measure,
            hw_efficient=hw_efficient,
            device=device,
        )

        # We'll still pick an entangler function
        self.entangler_forward = {
            "basic": self.bel_forward,
            "strong": self.sel_forward,
        }[entangling]

        # initialize weights
        weights_per_layer = 3 if entangling == "strong" else 1
        self.weights = nn.Parameter(
            torch.rand(1, depth, weights_per_layer, self.n_qubits) * 2 * torch.pi
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        1) Normalize x to encode it as amplitude vector
        2) Repeat entangling
        3) Measure
        """
        batched = True
        if x.dim() == 1:
            # Insert a batch dimension of size 1
            x = x.unsqueeze(0)
            batched = False

        if batched:
            batch_size = x.shape[0]

        # 1) Amplitude embedding:
        #    shape => (batch_size, 2^n_qubits)
        #    after normalization, this *becomes* our state.
        # Note: `self.amplitude_embedding(...)` is defined in SpeedyLayer
        phi = self.amplitude_embedding(x).cfloat().to(x.device)

        # 2) Variational entangling layers
        eigenvals_w = (
            rz_eigenvals(self.weights, mask=self.binary_matrix)[0]
            .cfloat()
            .to(x.device)
        )
        for j in range(self.depth):
            phi = self.entangler_forward(phi, eigenvals_w[0, j])

        # 3) Measurement
        out = self.meas(phi)

        if batched:
            out = out.view(batch_size, -1)
        else:
            out = out.view(-1)

        return out

#############################################
#  (All other classes remain the same ...)  #
#############################################
#############################################