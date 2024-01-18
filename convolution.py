from jax import numpy as np

from typing import Tuple

def conv2d(input_data: np.ndarray, output_data: np.ndarray, filter: np.ndarray, bias: float, stride: int = 1) -> np.ndarray:
    """
    Perform 2D convolution operation on the input data (2D image) using the given filter and bias.
    Writes to provided output_data array.

    Args:
        input_data (np.ndarray): Input data of shape (input_height, input_width).
        output_data (np.ndarray): Output data of shape (output_height, output_width).
        filter (np.ndarray): Filter of shape (filter_height, filter_width).
        bias (float): Scalar bias.
        stride (int, optional): Stride value for the convolution operation. Defaults to 1.

    Returns:
        np.ndarray: Output tensor of shape (batch_size, output_height, output_width).

    """

    # Get dimensions of output data and filter
    output_height, output_width = output_data.shape

    filter_height, filter_width = filter.shape

    # Perform convolution
    for i in range(output_height):
        for j in range(output_width):
            # Extract the current patch from input data
            patch = input_data[i*stride:i*stride+filter_height, j*stride:j*stride+filter_width]

            # Perform element-wise multiplication and sum using Jax numpy primitives
            output_data = output_data.at[i, j].set(np.sum(patch * filter) + bias)

    return output_data

def conv2d_forward(input_data: np.ndarray, filter: np.ndarray, bias: float, stride: int = 1, padding: int = 0) -> np.ndarray:
    """
    Perform forward convolution operation on the input data (2D images) using the given filter and bias.

    Args:
        input_data (np.ndarray): Input data of shape (batch_size, input_height, input_width).
        filter (np.ndarray): Filter of shape (filter_height, filter_width).
        bias (np.ndarray): Bias of shape (1, 1).
        stride (int, optional): Stride value for the convolution operation. Defaults to 1.
        padding (int, optional): Padding value for the input data. Defaults to 0, assumed to be symmetric.

    Returns:
        np.ndarray: Output tensor of shape (batch_size, output_height, output_width).
    """
    # Get dimensions of input data and filter
    batch_size, input_height, input_width = input_data.shape
    filter_height, filter_width = filter.shape

    # Calculate output dimensions
    output_height = (input_height - filter_height + 2 * padding) // stride + 1
    output_width = (input_width - filter_width + 2 * padding) // stride + 1

    # Add padding to input data.
    # Assuming odd filter dimensions here in case of non-zero padding.
    padded_input = np.pad(input_data, pad_width=((0, 0), (padding, padding), (padding, padding)), mode='constant')

    # Initialize output tensor
    output = np.zeros((batch_size, output_height, output_width))

    # Perform convolution
    for b in range(batch_size):
        output = output.at[b].set(conv2d(padded_input[b], output[b], filter, bias, stride))

    return output

def dilate_2d_arrays(input_data: np.ndarray, dilution_count: int) -> np.ndarray:
    """
    Dilate the each input 2D arrays by inserting dilution_count zeros between elements.

    Args:
        input_data (np.ndarray): Input array.
        dilution_count (int): Dilution count.

    Returns:
        np.ndarray: dilated array.
    """
    batch_size, input_height, input_width = input_data.shape
    output_height = (input_height - 1) * dilution_count + input_height
    output_width = (input_width - 1) * dilution_count + input_width
    step = dilution_count + 1

    output = np.zeros((batch_size, output_height, output_width))

    for b in range(batch_size):
        for i in range(input_height):
            for j in range(input_width):
                output = output.at[b, i * step, j * step].set(input_data[b, i, j])

    return output


def conv2d_backward(next_layer_grad: np.ndarray, input_data: np.ndarray, filter: np.ndarray, stride: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute the gradients of the input data and filter for the backward pass of a 2D convolution operation.

    It is a simplified version that uses the dilution trick to compute the gradients of the input data and filter as it allows
    to reuse the forward pass code.

    Args:
        next_layer_grad (np.ndarray): Gradient of the loss with respect to the output of the convolution layer.
        input_data (np.ndarray): Input data of shape (batch_size, input_height, input_width).
        filter (np.ndarray): Filter of shape (filter_height, filter_width). Only square filters are supported.
        stride (int, optional): Stride value for the convolution operation. Defaults to 1.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Gradients of the input data and filter.
    """
    batch_size, _, _ = input_data.shape
    filter_height, filter_width = filter.shape

    # dilate the gradient of the next layer by (stride - 1) - for filter gradient computation
    dilated_next_layer_grad = dilate_2d_arrays(next_layer_grad, stride - 1)

    # further pad the gradient of the next layer with (filter_height - 1) - for input data gradient computation
    padded_dilated_next_layer_grad = np.pad(dilated_next_layer_grad, 
                                        pad_width=((0, 0), (filter_height - 1, filter_height - 1), (filter_width - 1, filter_width - 1)), 
                                        mode='constant')

    # Input grad has the same shape as input
    input_grad = np.zeros(input_data.shape)
    # Filter grad has the same shape as filter, one for each sample in the batch
    filter_grad = np.zeros((batch_size, filter_height, filter_width))

    # Rotate the filter by 180 degrees
    rotated_filter = np.flip(np.flip(filter, axis=0), axis=1)

    for b in range(batch_size):
        # Gradient of the input is the "full convolution" of the rotated filter with the padded dilated gradient of the next layer
        input_grad = input_grad.at[b].set(conv2d(padded_dilated_next_layer_grad[b], input_grad[b], rotated_filter, 0, 1))       

        # Gradient of the filter is the "valid convolution" of the input data with the dilated gradient of the next layer
        filter_grad = filter_grad.at[b].set(conv2d(input_data[b], filter_grad[b], dilated_next_layer_grad[b], 0, 1))

    return input_grad, filter_grad

