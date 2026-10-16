import copy

import numpy as np
import torch


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_sum, correct, n = 0.0, 0, 0

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.squeeze(1).long().to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        n += targets.size(0)

    return correct / n * 100, loss_sum / n


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    """Returns (accuracy %, mean loss, true labels, class probabilities)."""
    model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    y_true, y_prob = [], []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.squeeze(1).long().to(device)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss_sum += loss.item() * targets.size(0)
        correct += (outputs.argmax(1) == targets).sum().item()
        n += targets.size(0)
        y_true.append(targets.cpu().numpy())
        y_prob.append(outputs.softmax(1).cpu().numpy())

    return correct / n * 100, loss_sum / n, np.concatenate(y_true), np.concatenate(y_prob)


class EarlyStopping:
    """Stops training when validation loss has not improved for `patience` epochs.

    `checkpoint_path` writes the best weights to disk every time they improve, so
    killing a run does not throw the training away.
    """

    def __init__(self, patience=5, min_delta=0.0, checkpoint_path=None):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.best_loss = np.inf
        self.best_epoch = 0
        self.best_state = None
        self.counter = 0
        self.stop = False

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            if self.checkpoint_path:
                torch.save({"epoch": epoch, "val_loss": val_loss,
                            "state_dict": self.best_state}, self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model


def fit(model, train_loader, val_loader, criterion, optimizer, device,
        n_epochs, patience=5, scheduler=None, checkpoint_path=None):
    """Trains with early stopping. Returns the history dict; model is left at its best weights."""
    stopper = EarlyStopping(patience=patience, checkpoint_path=checkpoint_path)
    history = {"train_acc": [], "train_loss": [], "val_acc": [], "val_loss": []}

    for epoch in range(1, n_epochs + 1):
        train_acc, train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc, val_loss, _, _ = evaluate(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(val_loss) if _needs_metric(scheduler) else scheduler.step()

        history["train_acc"].append(train_acc)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_loss"].append(val_loss)

        print(f"Epoch [{epoch}/{n_epochs}]  "
              f"train_loss {train_loss:.4f}  train_acc {train_acc:.2f}%  "
              f"val_loss {val_loss:.4f}  val_acc {val_acc:.2f}%")

        if stopper.step(val_loss, model, epoch):
            print(f"Early stopping at epoch {epoch}. Best epoch: {stopper.best_epoch} "
                  f"(val_loss {stopper.best_loss:.4f})")
            break

    stopper.restore(model)
    history["best_epoch"] = stopper.best_epoch
    return history


def _needs_metric(scheduler):
    return isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
