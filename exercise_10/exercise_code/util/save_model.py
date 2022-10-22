"""Utils for model saving"""

import os
import pickle
import torch

def save_model(model, file_name, directory="models"):
    """Save model as pickle"""
    model = model.cpu()
    if not os.path.exists(directory):
        os.makedirs(directory)
    model_path = os.path.join(directory, file_name)
    torch.save(model, model_path)
    return model_path
