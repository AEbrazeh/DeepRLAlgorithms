import numpy as np

def onehotEncode(input):
    
    '''
    One-hot encodes the input array.

    Parameters:
    input (array): The input array to be one-hot encoded.

    Returns:
    array: The one-hot encoded array.
    '''
    
    l = len(input)
    output = np.zeros((l, input.max()+1), dtype=bool)
    output[np.arange(l), input] = 1
    return output
