#!/usr/bin/env python3
# coding: utf-8
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
# limitations under the License.

"""repository_gpu"""
__all_gpu__ = {
    '2.Mul11.Mul1.TensorAdd1.Mul1.Mul45.Mul1.TensorAdd17.Mul1.Tanh1.Mul11.Sub1.Mul12.Mul12.Mul110.TensorAdd6.Mul1.TensorAdd13.Mul119.19': {
        '8192_3072--': {
            'float16--': {'dim': '0 0 1024 1024', 'bind_block': '12288', 'bind_thread':'1024'},
        },
    },
    '2.Cast1.TensorAdd13.Mul11.Mul12.Mul1.TensorAdd14.Mul1.Tanh1.TensorAdd1.Mul18.Mul1.312': {
        'metadata': {
            'attrs': {
                'enable_auto_fuse': False,
            },
        },
        '8192_3072.3072.8192_3072-': {
            'float16.float32.float16-': {"dim": '0 0 4 4 0 1 1024 1024', "bind_block": '3 1024 1', "bind_thread": '1024 1 1'},
        },
    },
    '3.Mul2.Mul14.Mul13.ReduceSum1.46': {
        '64_12_128_128---.64_12_128_1': {
            'float16----': {"dim":'0 0 128 128 0 1 128 128', "bind_block": '1 768 1', "bind_thread":'32 32 1'},
        },
    },
    '2.Cast2.TensorAdd12.Reshape1.Transpose1.5': {
        '768.8192_768.64_12_128_64': {
            'float32.float16-': {"dim":'0 0 1 1 0 1 1 1 0 2 16 16 0 3 64 64', "bind_block": '1 12 64', "bind_thread": '64 16 1'},
        },
    },
}