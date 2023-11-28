"""
- Implement a learning rate scheduler to automatically adjust the learning rate.
- The scheduler will check if the validation loss does not decrease for a given number of epochs
then decrease the learning rate by a given 'factor'.
"""

from torch.optim import lr_scheduler

class LRScheduler: 
    
 
    def __init__(self, optimizer, patience=5, min_lr=1e-6, factor=0.5):
        """
        :param optimizer: the optimizer we are using.
        :param patience: the number of epochs to wait before updating the learning rate.
        :param min_lr: the minimum learning rate. 
        :param factor: factor by which the learning rate should be updated.
        :returns:  new_lr = old_lr * factor.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.min_lr = min_lr
        self.factor = factor
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", patience=self.patience,
            factor=self.factor, min_lr=self.min_lr, verbose=True
        )

    # Treat the object as a function
    # Update scheduler everytime we have the validation loss. 
    def __call__(self, val_loss):
        self.lr_scheduler.step(val_loss)