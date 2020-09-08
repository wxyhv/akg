import numpy as np

import akg.tvm as tvm
import akg.topi as topi

def relu_grad_np(head, in_data):
    return np.where(in_data > 0, head, 0)

def bn_beta_grad_np(head, layout='NHWC'):
    if layout == 'NHWC':
        n, h, w, c = head.shape
    elif layout == 'NCHW':
        n, c, h, w = head.shape
    else:
        raise NotImplementedError(
            'layout is not supported {} '.format(layout)
        ) 
    
    reshape = np.reshape(head, (n * h * w, c))
    bn_beta_grad = np.sum(reshape, axis=0)
    return bn_beta_grad

def bn_gamma_grad_np(head, in_data, data_sum, layout='NHWC'):
    if layout == 'NHWC':
        n, h, w, c = head.shape
    elif layout == 'NCHW':
        n, c, h, w = head.shape
    else:
        raise NotImplementedError(
            'layout is not supported {} '.format(layout)
        )

    mean = np.divide(data_sum, n * c * h)
    x_hat = np.subtract(in_data, data_sum)
    x_hat_mul = np.multiply(x_hat, head)
    bn_gamma_grad = np.sum(np.reshape(x_hat_mul, (n * h * w, c)), axis=0)
    return bn_gamma_grad

    