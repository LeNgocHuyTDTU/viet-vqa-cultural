import os
import torch

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path='checkpoints/best_model.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch, val_loss)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(model, epoch, val_loss)
            self.counter = 0

    def save_checkpoint(self, model, epoch, val_loss):
        os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
        }, self.checkpoint_path)
        print(f'Validation loss giảm. Đã lưu Checkpoint tại epoch {epoch}!')