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

"""build module"""
import os
import json
import akg
from akg import tvm
from akg.tvm import _api_internal
from akg.topi.cuda.injective_single_kernel import schedule_injective
from .repository import __all__ as repository
from .repository_gpu import __all_gpu__ as repository_gpu
import topi

def generate_trait(desc):
    """ generate trait of kernel description """
    def generate_compute_trait():
        tensor_idx = {}
        counter = 0
        traits = []
        if desc['input_desc'] is not None:
            for in_desc in desc['input_desc']:
                tensor_idx[in_desc[0]['tensor_name']] = counter
                counter += 1
            traits = [str(len(desc['input_desc']))]
        for op in desc['op_desc'] if desc['op_desc'] is not None else []:
            input_idx = []
            for input_desc in op['input_desc']:
                if input_desc[0].get('value', None) is None:
                    input_idx.append(counter - tensor_idx[input_desc[0]['tensor_name']])
            input_idx.sort()
            input_idx_str = ''.join([str(i) for i in input_idx])
            traits.append(op['name'] + input_idx_str)
            tensor_idx[op['output_desc'][0]['tensor_name']] = counter
            counter += 1
        output_idx = []
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            output_idx.append(tensor_idx[out_desc['tensor_name']])
        output_idx.sort()
        traits.append(''.join([str(i) for i in output_idx]))
        return '.'.join(traits)

    def append_trait(traits, data):
        if traits and traits[-1].rstrip('-') == data:
            traits[-1] += '-'
        else:
            traits.append(data)

    def generate_shape_trait():
        traits = []
        for in_desc in desc['input_desc'] if desc['input_desc'] is not None else []:
            shape_s = '_'.join([str(i) for i in in_desc[0]['shape']])
            append_trait(traits, shape_s)
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            shape_s = '_'.join([str(i) for i in out_desc['shape']])
            append_trait(traits, shape_s)
        return '.'.join(traits)

    def generate_dtype_trait():
        traits = []
        for in_desc in desc['input_desc'] if desc['input_desc'] is not None else []:
            dtype = in_desc[0]['data_type']
            append_trait(traits, dtype)
        for out_desc in desc['output_desc'] if desc['output_desc'] is not None else []:
            dtype = out_desc['data_type']
            append_trait(traits, dtype)
        return '.'.join(traits)

    compute = generate_compute_trait()
    shape = generate_shape_trait()
    dtype = generate_dtype_trait()
    return compute, shape, dtype

def _build_to_func(desc_s, desc_d, attr=None):
    """
    build kernel with compute description in json format
    Args:
       desc_s : str of compute description
       desc_d : dict of compute description
       attr   : dict of build attributes

    Returns:
       Module.
    """
    def get_repo(keys, default=None):
        repo = repository
        for key in keys:
            repo = repo.get(key)
            if not repo:
                return default
        return repo
    if attr is None:
        attr = {'dim': ''}
    # turn 'enable_auto_inline' off for composite op by default.
    if 'enable_auto_inline' not in attr:
        attr['enable_auto_inline'] = False
    compute, shape, dtype = generate_trait(desc_d)
    repo_attr = get_repo([compute, shape, dtype, 'metadata', 'attrs'], {})
    if not repo_attr:
        repo_attr = get_repo([compute, 'metadata', 'attrs'], {})
    for a in repo_attr:
        if not attr.get(a):
            attr[a] = repo_attr[a]
    if attr.get('dim') in (None, ''):
        tiling = get_repo([compute, shape, dtype, 'dim'])
        if tiling:
            attr['dim'] = tiling
    func = tvm.get_global_func("composite_with_json_to_func")
    return func(desc_s, attr)

def _build_to_gpu_func(desc_s, desc_d, attr=None, poly=False):
    """
    build kernel with compute description in json format
    Args:
       desc_s : str of compute description
       desc_d : dict of compute description
       attr   : dict of build attributes

    Returns:
       Module.
    """
    def get_repo(keys, default=None):
        repo = repository_gpu
        for key in keys:
            repo = repo.get(key)
            if not repo:
                return default
        return repo
    if attr is None:
        attr = {'dim': ''}
    compute, shape, dtype = generate_trait(desc_d)
    repo_attr = get_repo([compute, shape, dtype, 'metadata', 'attrs'], {})
    if not repo_attr:
        repo_attr = get_repo([compute, 'metadata', 'attrs'], {})
    for a in repo_attr:
        if not attr.get(a):
            attr[a] = repo_attr[a]
    attr_list = ['dim', 'bind_block', 'bind_thread']
    for item in attr_list:
        if attr.get(item) in (None, ''):
            value = get_repo([compute, shape, dtype, item])
            if value:
                attr[item] = value
    func = tvm.get_global_func("composite_with_json")
    return func(desc_s, attr, poly)

def _build(desc_s, desc_d, attrs=None, poly=False):
    if desc_d['process'] == 'cuda':
        return _build_to_gpu_func(desc_s, desc_d, attrs, poly)
    rst = _build_to_func(desc_s, desc_d, attrs)
    return _api_internal._BuildToModule(rst)

def build(kernel_desc, attrs=None, poly=False):
    """
    build kernel with compute description in json format
    Args:
       kernel_desc : str or dict of compute description
       attrs   : dict of build attributes

    Returns:
       Module.
    """
    if isinstance(kernel_desc, str):
        desc_s = kernel_desc
        desc_d = json.loads(kernel_desc)
    else:
        assert isinstance(kernel_desc, dict)
        desc_s = json.dumps(kernel_desc)
        desc_d = kernel_desc
    return _build(desc_s, desc_d, attrs, poly)

def get_tiling_space(kernel_desc, level=1, attr=None):
    """
    get tiling space of composite kernel
    Args:
       kernel_desc : str of compute description
       level       : info level
       attr        : dict of build attributes

    Returns:
       Module.
    """
    if attr is None:
        attr = {}
    attr['help_tiling'] = level
    attr['tuning'] = 'on'
    func = tvm.get_global_func('composite_lower')
    ret = func(kernel_desc, attr)
    spaces = {}
    spaces['index'] = ret.index_table.asnumpy().tolist()
    spaces['l1_range'] = ret.l1_tile_range_table.asnumpy().tolist()
    spaces['l0_range'] = ret.l0_tile_range_table.asnumpy().tolist()
    spaces['l1_mod'] = ret.l1_tile_mod_table.asnumpy().tolist()
    spaces['l0_mod'] = ret.l0_tile_mod_table.asnumpy().tolist()
    if level >= 2:
        spaces['tuning_space'] = ret.tiling_candidate.asnumpy().tolist()
    return spaces

@tvm.register_func("akg_build_gpu_module")
def build_cuda(outputs, args, sch_name, kernel_name, attrs = False, poly = False, binds = None):
    s = select_cuda_scheduler(outputs, sch_name, poly)
    if attrs:
        attrs_t = dict(attrs.items())
    else:
        attrs_t = None
    dump_ir = os.getenv('MS_AKG_DUMP_IR') == "on"
    with tvm.build_config(dump_pass_ir = dump_ir):
        mod = akg.build(s, list(args), "cuda", name = kernel_name, binds = binds, attrs = attrs_t, polyhedral=bool(poly))
        return mod

@tvm.register_func("select_cuda_scheduler")
def select_cuda_scheduler(outputs, sch_name, poly = False):
    scheduler = {
        "injective" : topi.cuda.injective_single_kernel.schedule_injective,
        "reduce"    : topi.cuda.reduce_opt.schedule_reduce,
    }
    with tvm.target.cuda():
        if bool(poly):
            s = akg.tvm.create_schedule([x.op for x in list(outputs)])
        else:
            s = scheduler[sch_name](outputs)
        return s
