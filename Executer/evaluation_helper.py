import torch
from torch import nn
from torch.utils.data import DataLoader


def calc_entropy_loss_batch(input_batch: torch.Tensor, target_batch: torch.Tensor, model: nn.Module, device: torch.device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)

    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )
    return loss


def eval_entropy_loss_batch(data_loader: DataLoader, model: nn.Module, device: torch.device):
    model.eval()
    loss_count = 0
    loss = 0

    # Use the whole data loader as evaluation step
    with torch.no_grad():
        for input_batch, target_batch in data_loader:
            loss += calc_entropy_loss_batch(input_batch, target_batch, model, device)
            loss_count += 1

    model.train()

    return loss / loss_count
