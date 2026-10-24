"""Complete educational MLP. No NumPy, autograd, sklearn or third-party imports.

Weights have shape [output][input], preserving the original repo's convention.
The NumPy engine is a batched implementation of these same equations.
"""
import math
import random


def softmax(logits):
    maximum = max(logits)
    values = [math.exp(z - maximum) for z in logits]
    total = sum(values)
    return [v / total for v in values]


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


class ReferenceMLP:
    def __init__(self, dims=(784, 20, 10), activation="relu", seed=42):
        if activation not in ("relu", "sigmoid") or len(dims) < 2 or min(dims) < 1:
            raise ValueError("Invalid architecture")
        self.dims, self.activation = list(dims), activation
        rng = random.Random(seed)
        self.weights = []
        self.biases = []
        for i, (nin, nout) in enumerate(zip(dims[:-1], dims[1:])):
            std = math.sqrt((2.0 if activation == "relu" and i < len(dims)-2 else 1.0) / nin)
            self.weights.append([[rng.gauss(0.0, std) for _ in range(nin)] for _ in range(nout)])
            self.biases.append([0.0] * nout)

    def forward(self, x):
        if len(x) != self.dims[0]:
            raise ValueError("Input dimension mismatch")
        acts, preacts = [list(x)], []
        for l, (weights, bias) in enumerate(zip(self.weights, self.biases)):
            previous = acts[-1]
            z = [sum(w * a for w, a in zip(row, previous)) + b for row, b in zip(weights, bias)]
            preacts.append(z)
            if l == len(self.weights)-1:
                a = softmax(z)
            elif self.activation == "relu":
                a = [max(0.0, v) for v in z]
            else:
                a = [sigmoid(v) for v in z]
            acts.append(a)
        return acts, preacts

    def loss_and_grad(self, x, target):
        acts, zs = self.forward(x)
        maximum = max(zs[-1])
        loss = maximum + math.log(sum(math.exp(z-maximum) for z in zs[-1])) - zs[-1][target]
        delta = acts[-1].copy()
        delta[target] -= 1.0
        grad_w, grad_b = [None]*len(self.weights), [None]*len(self.weights)
        for l in reversed(range(len(self.weights))):
            grad_w[l] = [[d*a for a in acts[l]] for d in delta]
            grad_b[l] = delta.copy()
            if l:
                previous = [sum(self.weights[l][j][i]*delta[j] for j in range(len(delta)))
                            for i in range(self.dims[l])]
                if self.activation == "relu":
                    delta = [d if z > 0 else 0.0 for d,z in zip(previous,zs[l-1])]
                else:
                    delta = [d*a*(1-a) for d,a in zip(previous,acts[l])]
        return loss, grad_w, grad_b

    def train_one(self, x, y, learning_rate=0.03):
        loss, gw, gb = self.loss_and_grad(x, y)
        # Every gradient is computed before any parameter is updated.
        for l in range(len(self.weights)):
            for j, row in enumerate(self.weights[l]):
                for i in range(len(row)):
                    row[i] -= learning_rate * gw[l][j][i]
                self.biases[l][j] -= learning_rate * gb[l][j]
        return loss

    def predict(self, x):
        return max(enumerate(self.forward(x)[0][-1]), key=lambda p:p[1])[0]
