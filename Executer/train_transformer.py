import tiktoken
import torch
from torch import nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from Data.TextDataset import get_data_loader
from Executer.evaluation_helper import calc_entropy_loss_batch, eval_entropy_loss_batch
from Transformer.transformer_components import TransformerConfig, Transformer

train_paths = ['./assets/lovecraftcorpus/beast.txt', './assets/lovecraftcorpus/clergyman.txt', './assets/lovecraftcorpus/hound.txt']
validation_paths = ['./assets/lovecraftcorpus/beast.txt']

def get_full_text_from_path(path: str):
    with open(path, 'r') as file:
        file_content = file.read()
        return file_content

def start_train_process():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # A
    device = torch.device("mps") if torch.backends.mps.is_available() else device

    tokenizer = tiktoken.get_encoding("o200k_base")

    batch = 16
    config = TransformerConfig(
        embed_dim=768, window_size=50, vocab_size=tokenizer.n_vocab,
        attention_head_size=8, attention_layer_size=12, hidden_dropout_prob=0.1,
        inference_mode=True, device=device
    )

    train_loader = get_data_loader(train_paths, tokenizer, config.window_size, stride=8, batch_size=batch, drop_last=True)
    validation_loader = get_data_loader(validation_paths, tokenizer, config.window_size, stride=8,
                                        batch_size=batch, drop_last=False)
    model = Transformer(config)
    model = model.to(device)

    train(train_loader, validation_loader, model, config, num_epochs=4)


def train(train_loader: DataLoader, validation_loader: DataLoader, model: nn.Module, config: TransformerConfig,
          num_epochs: int):
    optimizer: Optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    eval_freq = 10

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_entropy_loss_batch(input_batch, target_batch, model, config.device)
            train_losses.append(loss)

            loss.backward()
            optimizer.step()

            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                # Use the whole data loader as evaluation step
                val_loss = eval_entropy_loss_batch(validation_loader, model, config.device)
                val_losses.append(val_loss)

                print(f"Ep {epoch + 1} (Step {global_step:06d}): "
                      f"Train loss {loss:.3f}, Val loss {val_loss:.3f}")

if __name__ == '__main__':
    start_train_process()