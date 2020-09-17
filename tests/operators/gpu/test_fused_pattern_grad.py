# Copyright 2020 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License

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

    