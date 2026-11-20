import torch

def train_tiny_micrograd_mlp(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    weights: list[torch.Tensor],
    biases: list[torch.Tensor],
    learning_rate: float,
    steps: int
) -> tuple:

    trained_weights = [w.clone() for w in weights]
    trained_biases = [b.clone() for b in biases]

    loss_history = []

    def forward(x):
        activations = [x]

        for w, b in zip(trained_weights, trained_biases):
            x = torch.tanh(x @ w.T + b)
            activations.append(x)

        return x.squeeze(-1), activations

    for _ in range(steps):

        if inputs.shape[0] == 0:
            loss_history.append(inputs.new_zeros(()))
            continue

        predictions, activations = forward(inputs)

        errors = predictions - targets
        loss = (errors ** 2).sum()

        loss_history.append(loss.clone())

        delta = 2 * errors.unsqueeze(-1)

        n_layers = len(trained_weights)

        weight_grads = [None] * n_layers
        bias_grads = [None] * n_layers

        for layer in reversed(range(n_layers)):

            delta = delta * (
                1 - activations[layer + 1] ** 2
            )

            weight_grads[layer] = (
                delta.T @ activations[layer]
            )

            bias_grads[layer] = delta.sum(dim=0)

            if layer > 0:
                delta = delta @ trained_weights[layer]

        # Update all weights and biases simultaneously
        trained_weights = [
            w - learning_rate * gw
            for w, gw in zip(trained_weights, weight_grads)
        ]

        trained_biases = [
            b - learning_rate * gb
            for b, gb in zip(trained_biases, bias_grads)
        ]

    # Final predictions and loss
    predictions, _ = forward(inputs)

    final_loss = ((predictions - targets) ** 2).sum()

    return (
        predictions,
        final_loss,
        trained_weights,
        trained_biases,
        loss_history
    )
