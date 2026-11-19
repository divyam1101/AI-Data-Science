"""Single-layer perceptron with a learnable bias, trained with the perceptron learning rule."""

from __future__ import annotations

from collections.abc import Sequence


class Perceptron:
    def __init__(self, features: int) -> None:
        self.weights = [0.0] * features
        self.bias = 0.0

    def predict(self, x: Sequence[float]) -> int:
        return int(self.bias + sum(w * xi for w, xi in zip(self.weights, x, strict=True)) > 0)

    def train(self, inputs: Sequence[Sequence[float]], targets: Sequence[int],
              learning_rate: float = 0.1, max_epochs: int = 100) -> int | None:
        """Train until an epoch makes no mistakes.

        Returns the number of epochs needed, or ``None`` if it did not converge
        (guaranteed to converge only for linearly separable data, e.g. AND/OR but not XOR).
        """
        if len(inputs) != len(targets):
            raise ValueError("inputs and targets differ in length")
        for epoch in range(1, max_epochs + 1):
            mistakes = 0
            for x, y in zip(inputs, targets, strict=True):
                error = y - self.predict(x)
                if error:
                    mistakes += 1
                    self.weights = [w + learning_rate * error * xi for w, xi in zip(self.weights, x, strict=True)]
                    self.bias += learning_rate * error
            if mistakes == 0:
                return epoch
        return None
