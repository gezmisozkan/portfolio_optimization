from typing import List, Optional, Tuple, Type

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import os

from ..models.base_model import BaseModel

# Constants
LEARNING_RATE: float = 0.0001
LOSS_THRESHOLD: float = 0.0002


def validate_params(
    model: Optional[nn.Module], embedding_dim: Optional[int], tgt_context_sets: list
) -> None:
    if tgt_context_sets is None:
        raise ValueError("tgt_context_sets must be provided for training.")

    if model is None and embedding_dim is None:
        raise ValueError("If no model is provided, embedding_dim must be specified.")

    if model and embedding_dim:
        raise ValueError("Do not provide embedding_dim when a model is given.")


def initialize_model(
    model_class: Type[BaseModel], n_time_series: int, embedding_dim: int
) -> nn.Module:
    return model_class(n_time_series, embedding_dim)


def get_train_loader(tgt_context_sets: list, batch_size: int = 64) -> DataLoader:
    x_vals = np.array([idx[1] for idx in tgt_context_sets])
    y_vals = np.array([idx[0] for idx in tgt_context_sets])

    x_train_tensor = torch.from_numpy(x_vals).long()
    y_train_tensor = torch.from_numpy(y_vals).long()

    train_data = CustomDataset(x_train_tensor, y_train_tensor)
    # 8 workers with CUDA; on macOS each worker would copy the full tensors into RAM.
    num_workers = 8 if torch.cuda.is_available() else 0
    return DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)


def should_early_stop(losses: List[float], epoch: int) -> bool:
    return epoch > 2 and abs(losses[-1] - losses[-2]) < LOSS_THRESHOLD


def train_embeddings_from_tgt_context_sets(
    n_time_series: int,
    tgt_context_sets: Optional[list] = None,
    model: Optional[nn.Module] = None,
    epochs: int = 20,
    batch_size: int = 64,
    embedding_dim: Optional[int] = None,
    device: str = "cpu",
    early_stopping: bool = True,
    verbose: bool = True,
    save_checkpoint: bool = True,
    save_just_last_checkpoint: bool = False,
    checkpoint_folder= "checkpoints",
    tickers: Optional[list] = None,
) -> Tuple[nn.Module, List[float]]:
    validate_params(model, embedding_dim, tgt_context_sets)

    if model is None:
        model = initialize_model(BaseModel, n_time_series, embedding_dim)
    else:
        embedding_dim = model.embeddings.weight.shape[1]

    loss_function = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    train_loader = get_train_loader(tgt_context_sets, batch_size=batch_size)

    if verbose:
        print("Training embeddings...")

    if save_checkpoint:
        os.makedirs(checkpoint_folder, exist_ok=True)

    model, losses = train_model(
        epochs,
        model,
        loss_function,
        optimizer,
        train_loader,
        device,
        early_stopping,
        verbose,
        save_checkpoint=save_checkpoint,
        save_just_last_checkpoint=save_just_last_checkpoint,
        checkpoint_folder = checkpoint_folder,
        tickers=tickers,
    )

    return model, losses


def train_model(
    epochs: int,
    model: nn.Module,
    loss_function: nn.Module,
    optimizer: optim.Optimizer,
    train_loader: DataLoader,
    device: str = "cpu",
    early_stopping: bool = True,
    verbose: bool = True,
    save_checkpoint: bool = True,
    save_just_last_checkpoint: bool = False,
    checkpoint_folder: Optional[str] = None,
    tickers: Optional[list] = None,
) -> Tuple[nn.Module, List[float]]:
    """
    Train a PyTorch model.

    Parameters
    ----------
    epochs : int
        The number of epochs for training.
    model : nn.Module
        The PyTorch model to be trained.
    loss_function : nn.Module
        The loss function for training.
    optimizer : optim.Optimizer
        The optimizer for training.
    train_loader : DataLoader
        The data loader for training data.
    device : str, optional
        The device ('cuda', 'cpu', etc.) for training.
    verbose : bool, optional
        Flag to control verbosity of training process.

    Returns
    -------
    tuple
        The trained model and the list of loss values for each epoch.
    """
    model.to(device)
    losses = []

    for epoch in tqdm(range(epochs), disable=not verbose):
        total_loss = 0
        count = 0

        for x_batch, y_batch in tqdm(train_loader, desc=f"Batch {epoch}", disable=not verbose):
            count += len(y_batch)
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            log_probs = model(x_batch)
            loss = loss_function(log_probs, y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        epoch_loss = total_loss / count
        losses.append(epoch_loss)

        if verbose:
            print(f"Epoch {epoch}: Loss = {epoch_loss}")
    
        if save_just_last_checkpoint:
            checkpoint_path = os.path.join(checkpoint_folder, f"last_epoch.pt")
            model.save_model(checkpoint_path, tickers=tickers)
            print(f"Checkpoint saved at {checkpoint_path} for epoch {epoch}.")

        elif save_checkpoint:
            checkpoint_path = os.path.join(checkpoint_folder, f"epoch_{epoch}.pt")
            model.save_model(checkpoint_path, tickers=tickers)
        
        

        if early_stopping & should_early_stop(losses, epoch):
            if verbose:
                print(f"Early stopping at epoch {epoch} due to minimal loss reduction.")
            break

    return model, losses


class CustomDataset(Dataset):
    def __init__(self, x_tensor: torch.Tensor, y_tensor: torch.Tensor):
        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index: int) -> tuple:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)
