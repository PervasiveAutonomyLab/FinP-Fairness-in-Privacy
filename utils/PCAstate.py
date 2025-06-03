import numpy as np
import torch

def flatten(weight):
    '''
    Flatten the parameters of a model.

    Parameters:
    - weight: The model weights.

    Returns:
    - Flattened model weights.
    '''
    weight_flatten = []

    for param in weight.values():
        weight_flatten.append(np.array(param.cpu().detach()).reshape(-1))

    weight_flatten = [item for sublist in weight_flatten for item in sublist]

    return weight_flatten

def flatten_state(state_list):
    '''
    Flatten a list of states.

    Parameters:
    - state_list: List of states where each state is represented as a list of sublists.

    Returns:
    - Flattened list of states as a torch.Tensor.
    '''
    result_list = []
    max_length = max(len(sublist) for sublist in state_list)  # Find the maximum length of sublists

    for i in range(max_length):
        for sublist in state_list:
            if i < len(sublist):
                element = sublist[i]
                if isinstance(element, list):
                    result_list.extend(element)
                else:
                    result_list.append(element)

    return result_list
