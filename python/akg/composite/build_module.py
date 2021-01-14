#!/usr/bin/env python3
# coding: utf-8
# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from functools import reduce
import logging
import akg
from akg import tvm
from akg.tvm import _api_internal
from akg.topi.cuda.injective_single_kernel import schedule_injective
import topi

class Graph():
    def __init__(self, output):
        self.tensors = set(output)
        self.ops = []
        self.output_name = output
        self.input_name = []
        self.input = []
        self.core_num = 0
        self.output = []
        self.op_name = 'Fused'

def intersect(inp1, inp2):
    # compute intersect set of two lists.
    sort_inp1 = sorted(inp1)
    sort_inp2 = sorted(inp2)
    if (sort_inp1 == sort_inp2):
        return sort_inp1
    if not sort_inp1 and sort_inp2:
        return sort_inp2
    if not sort_inp2 and sort_inp1:
        return sort_inp1
    return list(set(sort_inp1).intersection(set(sort_inp2)))

def dominance_analysis(in_out_dict, start_node, stitch_node):
    # build dominance tree
    consumer_list = start_node + list(in_out_dict.keys())
    doms = [[]] * len(consumer_list)
    consumer_doms = dict(zip(consumer_list, doms))
    changed = True
    for node in start_node:
        consumer_doms[node].append(node)
    while changed:
        changed = False
        for node in consumer_list[len(start_node):]:
            new_idom = consumer_doms[in_out_dict[node][0]].copy()
            for predecessor in in_out_dict[node][1:]:
                # when predecessor dom is not empty, get the intersect set.
                if predecessor in consumer_doms and consumer_doms[predecessor]:
                    new_idom = intersect(consumer_doms[predecessor], new_idom)

            new_idom.append(node)
            if sorted(new_idom) != sorted(consumer_doms[node]):
                consumer_doms[node] = new_idom
                changed = True
    return consumer_doms

def clean_op_detect(in_out_dict, stitch_node):
    """
    For buffer stitch, detect fake outputs in json.
    These fake outputs would not be in alloc_map but stored in clean_op_map instead.
    """
    input_set = set(list(in_out_dict.keys()))
    clean_op_list = []
    for node in stitch_node:
        if node not in input_set:
            clean_op_list.append(node)
    return clean_op_list

def topology_analysis(in_out_dict):
    # topologically sorting graph nodes.
    topo_sort = []
    input_list = list(in_out_dict.keys())
    output_list = [out for out_list in in_out_dict.values() for out in out_list]
    output_set = set(output_list)
    vertice_set = set(input_list + output_list)
    # Difference set between output and all_vertices is nodes with 0-indegree.
    indegree_0 = vertice_set.difference(output_set)
    topo_sort.extend(list(indegree_0))
    while in_out_dict:
        next_node_list = []
        for node in indegree_0:
            if node in in_out_dict:
                next_node_list.extend(in_out_dict[node])
                # pop nodes
                in_out_dict.pop(node)
        indegree_0 = set()
        update_out_list = [out for out_list in in_out_dict.values() for out in out_list]
        for next_node in next_node_list:
            if next_node not in update_out_list:
                indegree_0.add(next_node)
        topo_sort.extend(list(indegree_0))
    return topo_sort

def shared_memory_optimization(in_out_dict, consumer_doms, topo_sort, req_map):
    # algorithm for shared memory allocate and reuse.
    # Return: alloc_map: dict{op_name: ['ALLOC', size], op_name:[shared_op_name, size]}
    alloc_map = dict()
    reuse_map = dict()
    for inst in topo_sort:
        if inst in req_map:
            shared = False
            for alloc in alloc_map:
                # when inst dom alloc, inst may reuse alloc memory.
                if alloc in consumer_doms and inst in consumer_doms[alloc]:
                    # rule: inst reuses alloc if inst size less equal than alloc size.
                    if req_map[inst] <= alloc_map[alloc][1]:
                        reuse_map[inst] = [alloc, req_map[inst]]
                        shared = True
                        break
            if not shared:
                alloc_map[inst] = ['ALLOC', req_map[inst]]
    return alloc_map, reuse_map

def parse_merged_json(desc_d, stitch_tensor_name, input_tensor_name, output_tensor_name):
    sub_graph_length = len(stitch_tensor_name) + 1
    sub_graph_node = [set() for _ in range(sub_graph_length)]
    extra_subgraph_output = dict(zip(range(sub_graph_length), [[] for _ in range(sub_graph_length)]))
    in_out_dict = {}
    inter_output_list = []
    idx = sub_graph_length - 1
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        op_info = desc_d['op_desc'][i]
        for out_desc in op_info['output_desc']:
            if out_desc['tensor_name'] in stitch_tensor_name:
                idx -= 1
            sub_graph_node[idx].add(out_desc['tensor_name'])
            for input_desc in op_info['input_desc']:
                for sub_input_desc in input_desc:
                    sub_graph_node[idx].add(sub_input_desc['tensor_name'])
                    if sub_input_desc['tensor_name'] in output_tensor_name:
                        inter_output_list.append(sub_input_desc['tensor_name'])
                    for subgraph in sub_graph_node[idx + 1:]:
                        tmp_name = sub_input_desc['tensor_name']
                        extra_output = tmp_name in subgraph and tmp_name not in stitch_tensor_name and tmp_name not in input_tensor_name
                        if extra_output:
                            extra_subgraph_output[idx].insert(0, sub_input_desc['tensor_name'])
                            break
                    if sub_input_desc['tensor_name'] not in in_out_dict:
                        in_out_dict[sub_input_desc['tensor_name']] = [out_desc['tensor_name']]
                    else:
                        in_out_dict[sub_input_desc['tensor_name']].append(out_desc['tensor_name'])
    final_output_list = [output for output in output_tensor_name if output not in inter_output_list]
    return in_out_dict, extra_subgraph_output, final_output_list

def collect_subgraph_info(desc_d, sub_stitch_graphs, req_map, input_tensor_name, output_tensor_name, stitch_node_list):
    inplace_assign_map = {}
    # traversal desc_d by reverse topologically order.
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        op_info = desc_d['op_desc'][i]
        if (op_info['name'] == "InplaceAssign"):
            inplace_assign_map[op_info['output_desc'][0]['tensor_name']] = op_info['input_desc'][0][0]['tensor_name']
        for sg in sub_stitch_graphs:
            for out_desc in op_info['output_desc']:
                out_tensor_name = out_desc['tensor_name']
                if out_tensor_name in sg.tensors:
                    sg.ops.append(op_info)
                    if out_tensor_name in req_map:
                        if out_desc['shape']:
                            req_map[out_tensor_name] = reduce(lambda x, y: x * y, out_desc['shape'])
                        else:
                            req_map[out_tensor_name] = 1

                    if out_tensor_name in sg.output_name:
                        sg.output.append(out_desc)

                    for input_desc in op_info['input_desc']:
                        for sub_input_desc in input_desc:
                            input_name = sub_input_desc['tensor_name']
                            if input_name in output_tensor_name:
                                sg.output.insert(0, sub_input_desc)
                            if input_name in input_tensor_name and input_name not in sg.input_name:
                                sg.input_name.append(sub_input_desc['tensor_name'])
                                sg.input.append([sub_input_desc])
                            # stop expand subgraph when encounter with stitch node.
                            if input_name not in stitch_node_list:
                                sg.tensors.add(sub_input_desc['tensor_name'])
                            # add extra input into subgraph.
                            elif input_name not in sg.output_name and input_name not in sg.input_name:
                                sg.input_name.append(input_name)
                                sg.input.append([sub_input_desc])
    return sub_stitch_graphs


def sub_graph_info(sub_graph, desc_d):
    # gather info for sub graph.
    op_json_str = {}
    op_json_str['composite'] = True
    op_json_str['composite_graph'] = desc_d['composite_graph']
    op_json_str['id'] = desc_d['id']
    op_json_str['op'] = sub_graph.op_name
    op_json_str['input_desc'] = sub_graph.input
    op_json_str['op_desc'] = sub_graph.ops
    op_json_str['output_desc'] = sub_graph.output
    op_json_str['platform'] = "AKG"
    op_json_str['process'] = desc_d['process']
    if 'sub_block_size' in desc_d['buffer_stitch']:
        op_json_str['blocksize'] = desc_d['buffer_stitch']['sub_block_size']

    json_str = json.dumps(op_json_str, indent=4)
    return json_str

def json_split(desc_d):
    """
    split sub graph from merged json file.
    Using 'buffer_stitch' to store stitch info from graph kernel.
    Args:
        desc_d: dict of compute description
    Returns:
        List of spilted json info.
        List of original input.
        Dict of dominance info.
    """
    stitch_jsons = []

    input_tensor_name = [tensor[0]['tensor_name'] for tensor in desc_d['input_desc']]
    output_tensor_name = [tensor['tensor_name'] for tensor in desc_d['output_desc']]
    stitch_node = desc_d['buffer_stitch']['stitch_op']
    stitch_node_name = [node for stitchnode in stitch_node for node in stitchnode]
    in_out_dict, extra_subgraph_output, final_output_list = parse_merged_json(desc_d, stitch_node_name, input_tensor_name, output_tensor_name)

    # traverse extra_subgraph_output to save extra output into subgraph
    for item in extra_subgraph_output:
        if extra_subgraph_output[item]:
            stitch_node[item] = extra_subgraph_output[item] + stitch_node[item]
    stitch_node_name = [node for stitchnode in stitch_node for node in stitchnode]

    # initialize req_map
    req_op_size = [0] * len(stitch_node_name)
    req_map = dict(zip(stitch_node_name, req_op_size))
    # add final output into stitch_op.
    stitch_node += [[op] for op in final_output_list]
    stitch_node_list = [node for stitchnode in stitch_node for node in stitchnode]

    # initialize sub_stitch_graphs.
    sub_stitch_graphs = []
    for i, stitch_op in enumerate(stitch_node):
        sub_stitch_graphs.append(Graph(stitch_op))

    # store InplaceAssign op output_tensor and first input_tensor.
    inplace_assign_map = {}

    sub_stitch_graphs = collect_subgraph_info(desc_d, sub_stitch_graphs, req_map, input_tensor_name, output_tensor_name, stitch_node_list)
    # reverse op order to generate topological subgraph
    for sg in sub_stitch_graphs:
        sg.ops = list(reversed(sg.ops))
        sg.op_name = desc_d['op']
        stitch_json_str = sub_graph_info(sg, desc_d)
        # print("====stitch_json====")
        # print(stitch_json_str)
        stitch_jsons.append(stitch_json_str)

    clean_op_list = clean_op_detect(in_out_dict, stitch_node_name)
    # add fake outputs into output_tensor_name
    output_tensor_name += clean_op_list
    dominance_dom = dominance_analysis(in_out_dict, output_tensor_name, stitch_node_name)
    topo_sort = topology_analysis(in_out_dict)
    alloc_map, reuse_map = shared_memory_optimization(in_out_dict, dominance_dom, topo_sort, req_map)
    # remove fake output from alloc_map and store them into clean_op_map
    clean_op_map = dict()
    for fake_op in clean_op_list:
        clean_info = alloc_map[fake_op] if fake_op in alloc_map else reuse_map[fake_op]
        clean_op_map[inplace_assign_map[fake_op]] = clean_info
        alloc_map.pop(fake_op) if fake_op in alloc_map else reuse_map.pop(fake_op)
    return stitch_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map


def split_json_to_graphs(desc_d):
    """
    spilt merge_json to single graph json.
    Args:
        desc_d : dict of compute desciption
    Returns:
        List of subgraph json.
        List of input names.
        Dict of output names.
    """
    op_jsons = []

    # get some basic info to init subgraph
    composite_graph_id = desc_d['composite_graph']
    composite_id = desc_d['id']
    final_output_name = desc_d['parallel_fusion']['sub_graph']
    sub_graphs = []
    for i in range(len(final_output_name)):
        sub_graphs.append(Graph(final_output_name[i]))

    # traversal desc_d by reverse topological order to construct subgraph
    for i in range(len(desc_d['op_desc']) - 1, -1, -1):
        op_info = desc_d['op_desc'][i]
        for g in sub_graphs:
            for j in range(len(op_info['output_desc'])):
                if op_info['output_desc'][j]['tensor_name'] in g.tensors:
                    g.ops.append(op_info)
                    for input_info in op_info['input_desc']:
                        for sub_input_info in input_info:
                            g.tensors.add(sub_input_info['tensor_name'])

    # get subgraph original input
    if desc_d['input_desc']:
        for op_input in desc_d['input_desc']:
            for g in sub_graphs:
                if op_input[0]['tensor_name'] in g.tensors:
                    g.input.append(op_input)

    # get subgraph original output
    for op_output in desc_d['output_desc']:
        for g in sub_graphs:
            if op_output['tensor_name'] in g.tensors:
                g.output.append(op_output)

    # get subgraph core num info
    core_num_info = desc_d['parallel_fusion']['core_num']
    for idx in range(len(sub_graphs)):
        g = sub_graphs[idx]
        g.core_num = core_num_info[idx]

    # reverse ops order to generate a topology order subgraph
    for g in sub_graphs:
        g.ops = list(reversed(g.ops))
        g.op_name = desc_d['op']

    # get the original input of all subgraphs in order
    # suppose all original json input_args info satisfies this order
    input_tensor_names = [tensor[0]['tensor_name'] for tensor in desc_d['input_desc']] if desc_d['input_desc'] else []
    output_tensor_names = [tensor['tensor_name'] for tensor in desc_d['output_desc']] if desc_d['output_desc'] else []

    # construct subgraph json info
    op_result = []
    for g in sub_graphs:
        op_json_str = {}
        op_json_str['composite'] = True
        op_json_str['composite_graph'] = composite_graph_id
        op_json_str['id'] = composite_id
        op_json_str['op'] = g.op_name
        op_json_str['input_desc'] = g.input
        op_json_str['op_desc'] = g.ops
        op_json_str['output_desc'] = g.output
        op_json_str['core_num'] = g.core_num
        op_json_str['platform'] = "AKG"
        op_json_str['process'] = desc_d['process']
        op_result.append(op_json_str)

    # all sub json info saved in op_jsons list
    for idx in range(len(op_result)):
        single_op = op_result[idx]
        json_str = json.dumps(single_op, indent=4)
        op_jsons.append(json_str)
    return op_jsons, input_tensor_names, output_tensor_names


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

def read_repo_file(repo_file):
    with open(repo_file, 'r') as f:
        repo = json.loads(f.read())
    return repo

def _get_repository_file_path(file):
    pwd = os.path.dirname(os.path.abspath(__file__))
    path = pwd + "/" + file
    if not os.path.exists(path):
        path = pwd + "/../config/" + file
        if not os.path.exists(path):
            raise FileNotFoundError("Can not find {} in directory {} and {}".format(file, pwd, pwd + "/../config"))
    return path

def _build_to_func(desc_s, desc_d, attr=None, use_repo=True):
    """
    build kernel with compute description in json format
    Args:
       desc_s : str of compute description
       desc_d : dict of compute description
       attr   : dict of build attributes

    Returns:
       Module.
    """
    if os.getenv('MS_GRAPH_KERNEL_TILING'):
        repository = read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
    else:
        file_path = _get_repository_file_path("repository.json")
        repository = read_repo_file(file_path)
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
    if use_repo:
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

def _reducemax_pattern(kernel_info):
    for op in kernel_info['op_desc']:
        if op['name'] == 'ReduceMax':
            input_shape = op['input_desc'][0][0]['shape']
            batch_size = input_shape[0]
            reduce_size = batch_size * input_shape[1] * input_shape[2]
            return (True, reduce_size)
    return (False, 0)

def _is_batchmatmul(kernel_info):
    for op in kernel_info['op_desc']:
        if op['name'] == 'BatchMatMul':
            return True
    return False

def _set_tiling_attrs(out_shape, attrs):
    axis_len = len(out_shape)
    if axis_len < 3:
        return attrs
    if all(map(lambda x:x == 1, [out_shape[x] for x in range(axis_len - 2)])):
        return attrs
    if attrs.get('bind_block') in (None, ''):
        i = 0
        while out_shape[i] == 1:
            i += 1
        block_y = out_shape[i] 
        block_x = out_shape[i + 1] if i < axis_len - 3 else 1
        attrs['bind_block'] = str(block_x) + ' ' + str(block_y)
    if attrs.get('dim') in (None, ''):
        batch_axis = 0
        for i in range(axis_len - 2):
            if out_shape[i] != 1:
                batch_axis += 1
        dim_list = [0, 0, 64, 64, 0, 0, 64, 64, 0, 0, 64, 4]
        dim_list = [0, 0, 1, 1] * batch_axis + dim_list
        i = 0
        while i < (len(dim_list) // 4):
            dim_list[i * 4 + 1] = i
            i += 1
        attrs['dim'] = ' '.join(str(x) for x in dim_list)
    return attrs

def _build_to_gpu_func(desc_s, desc_d, attrs=None, poly=False):
    """
    build kernel with compute description in json format
    Args:
       desc_s : str of compute description
       desc_d : dict of compute description
       attrs   : dict of build attributes

    Returns:
       Module.
    """
    if os.getenv('MS_GRAPH_KERNEL_TILING'):
        repository_gpu = read_repo_file(str(os.getenv('MS_GRAPH_KERNEL_TILING')))
    elif 'buffer_stitch' in desc_d:
        repository_gpu = {}
    else:
        file_path = _get_repository_file_path("repository_gpu.json")
        repository_gpu = read_repo_file(file_path)
    def get_repo(keys, default=None):
        repo = repository_gpu
        for key in keys:
            repo = repo.get(key)
            if not repo:
                return default
        return repo
    if attrs is None:
        attrs = {'dim': ''}
    compute, shape, dtype = generate_trait(desc_d)
    batchmatmul = _is_batchmatmul(desc_d)
    if batchmatmul:
        shape = "any_shape"
    repo_attr = get_repo([compute, shape, dtype, 'metadata', 'attrs'], {})
    if repo_attr and batchmatmul:
        repo_attr = _set_tiling_attrs(desc_d['output_desc'][0]['shape'], repo_attr)
    if not repo_attr:
        repo_attr = get_repo([compute, 'metadata', 'attrs'], {})
    for a in repo_attr:
        if not attrs.get(a):
            attrs[a] = repo_attr[a]
    attr_list = ['dim', 'bind_block', 'bind_thread']
    for item in attr_list:
        if attrs.get(item) in (None, ''):
            value = get_repo([compute, shape, dtype, item])
            if value:
                attrs[item] = value

    if 'parallel_fusion' in desc_d:
        block_jsons, input_tensor_names, output_tensor_names = split_json_to_graphs(desc_d)
        alloc_map, reuse_map, clean_op_map = dict(), dict(), dict()
        for i, _ in enumerate(block_jsons):
            if 'buffer_stitch' in block_jsons[i]:
                stitch_jsons, _, _, alloc_map, reuse_map, clean_op_map = json_split(block_jsons[i])
                block_jsons[i] = stitch_jsons
        func = tvm.get_global_func("composite_with_json_list_gpu")
        if not clean_op_map:
            clean_op_map['EMPTY'] = []
        if not reuse_map:
            reuse_map['EMPTY'] = []
        if not alloc_map:
            alloc_map['EMPTY'] = []

        return func(block_jsons, input_tensor_names, output_tensor_names, alloc_map, reuse_map, clean_op_map, attrs, poly)
    elif 'buffer_stitch' in desc_d:
        block_jsons = []
        stitch_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map = json_split(desc_d)
        block_jsons.append(stitch_jsons)
        func = tvm.get_global_func("composite_with_json_list_gpu")
        if not clean_op_map:
            clean_op_map['EMPTY'] = []
        if not reuse_map:
            reuse_map['EMPTY'] = []
        if _reducemax_pattern(desc_d)[0]:
            attrs['enable_tile_l0'] = True
            elem_per_thread = 4
            blockdim_x = 64
            blockdim_y = 16
            griddim_x = 1
            griddim_y = _reducemax_pattern(desc_d)[1] / (blockdim_y * elem_per_thread)
            attrs['dim'] = ' 0 0 128 64 0 1 128 128'
            attrs['bind_block'] = str(griddim_x) + ' ' + str(griddim_y)
            attrs['bind_thread'] = str(blockdim_x) + ' ' + str(blockdim_y)
        return func(block_jsons, input_tensor_name, output_tensor_name, alloc_map, reuse_map, clean_op_map, attrs, poly)
    func = tvm.get_global_func("composite_with_json")
    return func(desc_s, attrs, poly)

def _build(desc_s, desc_d, attrs=None, poly=False, use_repo=True):
    if desc_d['process'] == 'cuda':
        return _build_to_gpu_func(desc_s, desc_d, attrs, poly)
    rst = _build_to_func(desc_s, desc_d, attrs, use_repo)
    return _api_internal._BuildToModule(rst)

def build(kernel_desc, attrs=None, poly=False, use_repo=True):
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
    return _build(desc_s, desc_d, attrs, poly, use_repo)

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
    if 'enable_auto_inline' not in attr:
        attr['enable_auto_inline'] = False
    attr['pragma_reschedule'] = 1
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
def select_cuda_scheduler(outputs, sch_name, poly = False, grid_dims=0, block_dims=0, buffer_stitch=False):
    scheduler = {
        "injective" : topi.cuda.injective_single_kernel.schedule_injective,
        "reduce"    : topi.cuda.reduce_opt.schedule_reduce,
    }
    with tvm.target.cuda():
        if bool(poly):
            s = akg.tvm.create_schedule([x.op for x in list(outputs)])
        else:
            if grid_dims and block_dims and sch_name == "injective":
                s = scheduler[sch_name](outputs, grid_dims, block_dims, buffer_stitch=buffer_stitch)
            else:
                s = scheduler[sch_name](outputs, grid_dims, block_dims)
        return s
