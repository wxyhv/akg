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
# limitations under the License

import random
import numpy as np
from test_utils import gen_random_shape
from test_elem import test_elem
from test_elem_red import test_elem_red
from test_prim_pad import test_prim_pad
from test_prim_unpad import test_prim_unpad
from test_prim_transpose import test_prim_transpose
from test_concat_prim import test_concat_prim
from test_prim_argmax import test_prim_argmax
from test_prim_argmin import test_prim_argmin

def random_int_list(start, stop, length):
    start, stop = (int(start), int(stop)) if start <= stop else (int(stop), int(start))
    length = int(abs(length)) if length else 0
    random_list = []
    for i in range(length):
        random_list.append(random.randint(start, stop))
    return random_list

def elem(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        test_elem(fuzz_shape_lhs, fuzz_shape_rhs, "float32", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem(fuzz_shape_lhs, fuzz_shape_rhs, "float16", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem(fuzz_shape_lhs, fuzz_shape_rhs, "float32", multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem(fuzz_shape_lhs, fuzz_shape_rhs, "float16", multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
    else:
        pass

def elem_red(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        test_elem_red(fuzz_shape_lhs, fuzz_shape_rhs, "float32", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem_red(fuzz_shape_lhs, fuzz_shape_rhs, "float16", poly_sch=True, fusion_mode=fusion_mode)
        test_elem_red(fuzz_shape_lhs, fuzz_shape_rhs, "float32", axis=None, keepdims=False, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem_red(fuzz_shape_lhs, fuzz_shape_rhs, "float16", axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
    else:
        test_elem_red((32, 32, 32, 32), (32, 32, 32, 32), 'float32', axis=None, keepdims=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem_red((32, 32, 32, 32), (32, 1, 32, 1), 'float32', axis=None, keepdims=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem_red((4, 4, 4, 4), (4, 4, 4, 4), 'float16', axis=None, keepdims=False, poly_sch=True, fusion_mode=fusion_mode)
        test_elem_red((4, 4, 4, 4), (4, 1, 4, 1), 'float16', axis=None, keepdims=False, poly_sch=True, fusion_mode=fusion_mode)
        test_elem_red((32, 32, 32, 32), (32, 32, 32, 32), 'float32', axis=None, keepdims=False, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem_red((32, 32, 32, 32), (32, 1, 32, 1), 'float32', axis=None, keepdims=False, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_elem_red((4, 4, 4, 4), (4, 4, 4, 4), 'float16', axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
        test_elem_red((4, 4, 4, 4), (4, 1, 4, 1), 'float16', axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)

def prim_pad(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        length = len(fuzz_shape_lhs)
        start = 0
        stop = 10
        pad_before = random_int_list(start, stop, length)
        pad_after = random_int_list(start, stop, length)
        pad_value = random.uniform(start, stop)
        test_prim_pad(fuzz_shape_lhs, fuzz_shape_rhs, 'float32', pad_before, pad_after, pad_value=pad_value, multi_out=False, poly_sch=poly_sch, fusion_mode=fusion_mode) 
        test_prim_pad(fuzz_shape_lhs, fuzz_shape_rhs, 'float16', pad_before, pad_after, pad_value=pad_value, multi_out=False, poly_sch=True, fusion_mode=fusion_mode)
        test_prim_pad(fuzz_shape_lhs, fuzz_shape_rhs, 'float32', pad_before, pad_after, pad_value=pad_value, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode) 
        test_prim_pad(fuzz_shape_lhs, fuzz_shape_rhs, 'float16', pad_before, pad_after, pad_value=pad_value, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
    else:
        test_prim_pad((16, 16, 16, 16), (16, 1, 16, 1), 'float32', (1, 2, 3, 4), (4, 3, 2, 1), pad_value=2.0, multi_out=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_pad((16, 16, 16, 16), (16, 1, 16, 1), 'float16', (1, 2, 3, 4), (4, 3, 2, 1), pad_value=2.0, multi_out=False, poly_sch=True, fusion_mode=fusion_mode)
        test_prim_pad((16, 16, 16, 16), (16, 1, 16, 1), 'float32', (1, 2, 3, 4), (4, 3, 2, 1), pad_value=2.0, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_pad((16, 16, 16, 16), (16, 1, 16, 1), 'float16', (1, 2, 3, 4), (4, 3, 2, 1), pad_value=2.0, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)

def prim_unpad(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        length = len(fuzz_shape_lhs)
        start = 0
        stop = 10
        unpad_before = random_int_list(start, stop, length)
        pad_compare = np.greater(fuzz_shape_lhs, unpad_before)
        for i in range(len(unpad_before)):
            if pad_compare[i] == False:
                unpad_before[i] = fuzz_shape_lhs[i]-1

        unpad_after = random_int_list(start, stop, length)
        data_shape = np.subtract(fuzz_shape_lhs, unpad_before)
        pad_compare = np.greater(data_shape, unpad_after)
        for i in range(len(unpad_after)):
            if pad_compare[i] == False:
                unpad_after[i] = data_shape[i]-1
        print("unpad_before is ", unpad_before)
        print("unpad_after is ", unpad_after)
        test_prim_unpad(fuzz_shape_lhs, fuzz_shape_rhs, 'float32', unpad_before, unpad_after, multi_out=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_unpad(fuzz_shape_lhs, fuzz_shape_rhs, 'float16', unpad_before, unpad_after, multi_out=False, poly_sch=True, fusion_mode=fusion_mode)
        test_prim_unpad(fuzz_shape_lhs, fuzz_shape_rhs, 'float32', unpad_before, unpad_after, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_unpad(fuzz_shape_lhs, fuzz_shape_rhs, 'float16', unpad_before, unpad_after, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
    else:
        test_prim_unpad((16, 16, 16, 16), (16, 1, 16, 1), 'float32', (1, 2, 3, 4), (4, 3, 2, 1), multi_out=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_unpad((16, 16, 16, 16), (16, 1, 16, 1), 'float16', (1, 2, 3, 4), (4, 3, 2, 1), multi_out=False, poly_sch=True, fusion_mode=fusion_mode)
        test_prim_unpad((16, 16, 16, 16), (16, 1, 16, 1), 'float32', (1, 2, 3, 4), (4, 3, 2, 1), multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_unpad((16, 16, 16, 16), (16, 1, 16, 1), 'float16', (1, 2, 3, 4), (4, 3, 2, 1), multi_out=True, poly_sch=True, fusion_mode=fusion_mode)  

def concat_prim(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        concat_axis = 1 #when -b02, concat_axis=2 when -b13
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float32", concat_axis=concat_axis, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float16", concat_axis=concat_axis, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float32", concat_axis=concat_axis, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float16", concat_axis=concat_axis, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
    else:
        fuzz_shape_lhs = (32, 64, 128, 128)
        fuzz_shape_rhs = (1, 64, 1, 128)
        concat_axis = 1
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float32", concat_axis=concat_axis, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float16", concat_axis=concat_axis, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float32", concat_axis=concat_axis, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_concat_prim(fuzz_shape_lhs, fuzz_shape_rhs, "float16", concat_axis=concat_axis, multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)


def prim_transpose(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float32", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float16", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float32", multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float16", multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
    else:
        fuzz_shape_lhs = (32, 64, 128, 128)
        fuzz_shape_rhs = (1, 64, 1, 128)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float32", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float16", poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float32", multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_transpose(fuzz_shape_lhs, fuzz_shape_rhs, "float16", multi_out=True, poly_sch=poly_sch, fusion_mode=fusion_mode)

def prim_argmax(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        test_prim_argmax(fuzz_shape_lhs, fuzz_shape_rhs, "float32", axis=None, keepdims=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_argmax(fuzz_shape_lhs, fuzz_shape_rhs, "float16", axis=None, keepdims=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_argmax(fuzz_shape_lhs, fuzz_shape_rhs, "float32", axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
        test_prim_argmax(fuzz_shape_lhs, fuzz_shape_rhs, "float16", axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
    else:
        pass

def prim_argmin(poly_sch, fuzz_shape_lhs=None, fuzz_shape_rhs=None, fusion_mode=None):
    if fuzz_shape:
        test_prim_argmin(fuzz_shape_lhs, fuzz_shape_rhs, "float32", axis=None, keepdims=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_argmin(fuzz_shape_lhs, fuzz_shape_rhs, "float16", axis=None, keepdims=False, poly_sch=poly_sch, fusion_mode=fusion_mode)
        test_prim_argmin(fuzz_shape_lhs, fuzz_shape_rhs, "float32", axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
        test_prim_argmin(fuzz_shape_lhs, fuzz_shape_rhs, "float16", axis=None, keepdims=False, multi_out=True, poly_sch=True, fusion_mode=fusion_mode)
    else:
        pass


class Logger(object):
    def __init__(self, filename, stream):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def usage(op_map):
    print("Usage:")
    print("1. Run func1 and func2 with manual schedule:")
    print("\t$python test_all.py -m func_name1 func_name2")
    print("\t$python test_all.py --manual func_name1 func_name2")
    print("2. Run all with auto schedule:")
    print("\t$python test_all.py -a all\n")
    print("\t$python test_all.py --auto all\n")
    print("3. Both schedule methods will be tested if no option is specified")
    print("\t$python test_all.py func_name")
    print("4. Specify the fusion mode using options: ")
    print("\t$'-i' or '--input_hrz'', '-o' or '--output_hrz', '-d' or '--diamond', or depth fusion if not specified")
    print("4. Run fuzz test of add op with maximal dimension of shape equals 3")
    print("\t$python test_all.py -f 3 add")
    print("Available func:")
    print("\t", list(op_map.keys()), "\n")


if __name__ == '__main__':
    import sys
    import getopt
    import traceback
    from datetime import datetime

    op_map = {"elem": elem,
                            "elem_red": elem_red,
                            "prim_pad": prim_pad,
                            "prim_unpad": prim_unpad,
                            "prim_transpose": prim_transpose,
                            "concat_prim": concat_prim,
                            "prim_argmax": prim_argmax,
                            "prim_argmin": prim_argmin}
    all_f = list(op_map.values())
    op_map["all"] = all_f
    if len(sys.argv) == 1:
        usage(op_map)
        sys.exit()

    options, args = getopt.getopt(sys.argv[1:], "amf:b:iod", ["auto", "manual", "fuzz=", "broadcast=", "input_hrz", "output_hrz", "diamond"])
    schedule_method = 'both'
    fuzz_dim = 0
    broadcast_dim = ''
    fusion_mode = ''
    for name, value in options:
        if name in ("-a", "--auto"):
            schedule_method = "auto"
        if name in ("-m", "--manual"):
            schedule_method = "manual"
        if name in ("-f", "--fuzz"):
            fuzz_dim = int(value)
        if name in ("-b", "--broadcast"):
            broadcast_dim = str(value)

        if name in ("-i", "--input_hrz"):
            fusion_mode = "input_hrz"
        if name in ("-o", "--output_hrz"):
            fusion_mode = "output_hrz"
        if name in ("-d", "--diamond"):
            fusion_mode = "diamond"

    fail_op_list = dict()
    run_op_list = list()
    for op in args:
        if op_map.get(op) is not None:
            f = op_map.get(op)
            if not isinstance(f, list):
                run_op_list.append(f)
            else:
                run_op_list += f

    now = datetime.now()
    filename = "opstest_" + '-'.join(list(map(str, [now.month, now.day, now.hour, now.minute]))) + ".log"
    sys.stdout = Logger(filename, sys.stdout)
    sys.stderr = Logger(filename, sys.stderr)
    
    print("Schedule method: ", schedule_method)
    print("Fusion mode: ", fusion_mode)
    for op in run_op_list:
        print("Operater: ", op.__name__)
        fuzz_shape = gen_random_shape(fuzz_dim) if fuzz_dim > 0 else None
        broadcast_fuzz_shape = list(fuzz_shape) if fuzz_dim > 0 else None
        if fuzz_shape:
            print("Original fuzz shape: {}".format(fuzz_shape))
            for i in broadcast_dim:
                broadcast_fuzz_shape[int(i)] = 1
                print("Test broadcast fuzz shape: {}".format(broadcast_fuzz_shape))
    
        if schedule_method in ["both", "manual"]:
            try:
                print(" Time of manual schedule:")
                op(poly_sch=False, fuzz_shape_lhs=fuzz_shape, fuzz_shape_rhs=broadcast_fuzz_shape, fusion_mode=fusion_mode)
            except:
                if op.__name__ in fail_op_list:
                    fail_op_list[op.__name__].extend(["using manual schedule:", traceback.format_exc()])
                else:
                    fail_op_list[op.__name__] = ["using manual schedule:", traceback.format_exc()]

        if schedule_method in ["both", "auto"]:
            try:
                print("Time of auto schedule:")
                op(poly_sch=True, fuzz_shape_lhs=fuzz_shape, fuzz_shape_rhs=broadcast_fuzz_shape, fusion_mode=fusion_mode)
            except:
                if op.__name__ in fail_op_list:
                    fail_op_list[op.__name__].extend(["using auto schedule:", traceback.format_exc()])
                else:
                    fail_op_list[op.__name__] = ["using auto schedule:", traceback.format_exc()]

    if len(fail_op_list) == 0:
        print("All test pass!")
    else:
        for op, error_info in fail_op_list.items():
            print("Run op %s error"%op)
            for e in error_info:
                print(e)
