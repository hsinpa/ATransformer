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

    # Use the whole data loader as evaluation step
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)

    model.train()

    return loss
