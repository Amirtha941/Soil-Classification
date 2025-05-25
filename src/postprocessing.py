"""
Author: Annam.ai IIT Ropar  
Team Name: AgroMinds AI  
Team Members: Amirtha K, Tharun Babu C  
Leaderboard Rank: 41  
"""

# This module contains all the postprocessing logic used after model inference,
# including converting outputs to labels and evaluating performance.

import torch
import numpy as np
from sklearn.metrics import f1_score, classification_report

def get_predictions(outputs):
    """
    Converts model outputs (logits) into predicted class labels.
    """
    _, preds = torch.max(outputs, 1)
    return preds.cpu().numpy()

def evaluate_predictions(y_true, y_pred):
    """
    Computes F1 score and classification report for the predictions.
    """
    f1 = f1_score(y_true, y_pred, average='macro')
    min_f1 = np.min(f1_score(y_true, y_pred, average=None))
    report = classification_report(y_true, y_pred, digits=4)
    return {
        "macro_f1": f1,
        "min_f1": min_f1,
        "report": report
    }
