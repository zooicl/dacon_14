import numpy as np
import torch

from sklearn import metrics

def eval_summary(y_true, y_score, cut_off=0.5):
    if len(y_true) == 0 or len(y_score) == 0:
        return 'zero length'
    if len(y_true) != len(y_score):
        return 'diff length'
    
    y_pred = y_score.copy()
    y_pred[y_pred > cut_off] = 1
    y_pred[y_pred <= cut_off] = 0

    eval_dict = {}
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_score, pos_label=1)
    
    eval_dict['auc'] = metrics.auc(fpr, tpr)
    eval_dict['confusion_matrix'] = metrics.confusion_matrix(y_true, y_pred)
    
    pre, rec, _, _ = metrics.precision_recall_fscore_support(y_true, y_pred, pos_label=1)
    eval_dict['precision'] = pre[1]
    eval_dict['recall'] = rec[1]
    
    return eval_dict

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
#         self.improved = False

    def __call__(self, val_loss, model):

        score = -val_loss
#         self.improved = False

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
#             self.improved = True
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss