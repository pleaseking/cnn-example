import unittest
from jax import numpy as np

from convolution import conv2d_forward, conv2d_backward, dilate_2d_arrays

class ForwardTest(unittest.TestCase):
    def test_conv2d_forward(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        filter = np.array([[1, 1], [1, 1]])
        bias = 0
        stride = 1
        padding = 0
        expected_output = np.array([[[12, 16], [24, 28]]])
        output = conv2d_forward(input_data, filter, bias, stride, padding)
        self.assertTrue(np.array_equal(output, expected_output))

    def test_conv2d_forward_with_padding(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        bias = 0
        stride = 1
        padding = 1
        expected_output = np.array([[[12, 21, 16], [27, 45, 33], [24, 39, 28]]])
        output = conv2d_forward(input_data, filter, bias, stride, padding)
        self.assertTrue(np.array_equal(output, expected_output))

    def test_conv2d_forward_with_padding_and_bias(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        bias = 2
        stride = 1
        padding = 1
        expected_output = np.array([[[14, 23, 18], [29, 47, 35], [26, 41, 30]]])
        output = conv2d_forward(input_data, filter, bias, stride, padding)
        self.assertTrue(np.array_equal(output, expected_output))

    def test_conv2d_forward_with_stride(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        filter = np.array([[1, 1], [1, 1]])
        bias = 0
        stride = 2
        padding = 0
        expected_output = np.array([[[12]]])
        output = conv2d_forward(input_data, filter, bias, stride, padding)
        self.assertTrue(np.array_equal(output, expected_output))


class DilateTest(unittest.TestCase):
    def test_dilution_count_0(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        dilution_count = 0
        expected_output = input_data
        output = dilate_2d_arrays(input_data, dilution_count)
        self.assertTrue(np.array_equal(output, expected_output))

    def test_dilution_count_1(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        dilution_count = 1
        expected_output = np.array([[
            [1, 0, 2, 0, 3], 
            [0, 0, 0, 0, 0], 
            [4, 0, 5, 0, 6], 
            [0, 0, 0, 0, 0], 
            [7, 0, 8, 0, 9]]])
        output = dilate_2d_arrays(input_data, dilution_count)
        self.assertTrue(np.array_equal(output, expected_output))

    def test_dilution_count_2(self):
        input_data = np.array([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        dilution_count = 2
        expected_output = np.array([[
            [1, 0, 0, 2, 0, 0, 3], 
            [0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0], 
            [4, 0, 0, 5, 0, 0, 6], 
            [0, 0, 0, 0, 0, 0, 0], 
            [0, 0, 0, 0, 0, 0, 0], 
            [7, 0, 0, 8, 0, 0, 9]]])
        output = dilate_2d_arrays(input_data, dilution_count)
        self.assertTrue(np.array_equal(output, expected_output))

class BackwardTest(unittest.TestCase):
    def test_conv2d_backward(self):
        next_layer_grad = np.ones((1, 4, 3))
        input_data = np.array([[
            [1,1,1,2,3],
            [1,1,1,2,3],
            [1,1,1,2,3],
            [2,2,2,2,3],
            [3,3,3,3,3],
            [4,4,4,4,4]
        ]])
        filter = np.array([
            [1,0,-1],
            [2,0,-2],
            [1,0,-1]
        ])
        stride = 1
        expected_input_grad = np.array([[
            [ 1,  1,  0, -1, -1],
            [ 3,  3,  0, -3, -3],
            [ 4,  4,  0, -4, -4],
            [ 4,  4,  0, -4, -4],
            [ 3,  3,  0, -3, -3],
            [ 1,  1,  0, -1, -1]]])
        expected_filter_grad = np.array([[[15, 18, 25], [21, 23, 28], [30, 31, 34]]])
        input_grad, filter_grad = conv2d_backward(next_layer_grad, input_data, filter, stride)
        self.assertTrue(np.array_equal(input_grad, expected_input_grad))
        self.assertTrue(np.array_equal(filter_grad, expected_filter_grad))

if __name__ == '__main__':
    unittest.main()