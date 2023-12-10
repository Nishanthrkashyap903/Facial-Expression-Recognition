"""
- Implement the early stopping mechanism which breaks the training procedure 
when the loss does not improve over a certain number of iterations.
"""

class EarlyStopping: 
    def __init__(self, patience=10, min_delta=0):
        """
        :param patience: number of epochs to wait before stopping training. 
        :param min_delta: the minimum difference between the previous and the new loss to consider the network is improving.
        """
        self.best_loss = None # keep track of the current best loss. 
        self.min_delta = min_delta 
        self.patience = patience
        self.early_stop_enabled = False # a boolean value indicating if we should enable early stopping. 
        self.count = 0 # the number of epochs waiting. 
    
    def __call__(self, val_loss):
        # The first iteration, the best validation loss is None, update it. 
        if self.best_loss is None:
            self.best_loss = val_loss
        
        # The loss has been improved, update the best loss and reset the counter. 
        elif (self.best_loss - val_loss) > self.min_delta:
            self.best_loss = val_loss
            self.count = 0

        # If there is no improvement in the loss. 
        elif (self.best_loss - val_loss) < self.min_delta:
            self.count += 1
            print(f"[INFO] Early stopping: {self.count}/{self.patience}... \n\n")
            if self.count >= self.patience:
                self.early_stop_enabled = True
                print(f"[INFO] Early stopping enabled")