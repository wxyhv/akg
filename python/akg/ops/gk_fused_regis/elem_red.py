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

""" elem_red """
import akg
import akg.topi as topi
from akg.ops.gk_fused.elem_red import elem_red, elem_red_input_hrz, elem_red_output_hrz, elem_red_diamond

""" depth fusion """
@akg.schedule(topi.cuda.reduce_opt.schedule_reduce)
def elem_red_manual(data0, data1, axis=None, keepdims=False, multi_out=False):
    """ Fused operater: elem_red, with manual schedule """
    return elem_red(data0, data1, axis, keepdims, multi_out, poly_sch=False)

def elem_red_auto(data0, data1, axis=None, keepdims=False, multi_out=False):
    """Fused operater: elem_red, with auto schedule"""
    return elem_red(data0, data1, axis, keepdims, multi_out, poly_sch=True)

""" input-horizontal fusion """
@akg.schedule(topi.cuda.reduce_opt.schedule_reduce)
def elem_red_input_hrz_manual(data0, data1, axis=None, keepdims=False, multi_out=False):
    return elem_red_input_hrz(data0, data1, axis, keepdims, multi_out, poly_sch=False)

def elem_red_input_hrz_auto(data0, data1, axis=None, keepdims=False, multi_out=False):
    return elem_red_input_hrz(data0, data1, axis, keepdims, multi_out, poly_sch=True)

""" ouput-horizontal fusion """
@akg.schedule(topi.cuda.reduce_opt.schedule_reduce)
def elem_red_output_hrz_manual(data0, data1, axis=None, keepdims=False, multi_out=False):
    return elem_red_output_hrz(data0, data1, axis, keepdims, multi_out, poly_sch=False)

def elem_red_output_hrz_auto(data0, data1, axis=None, keepdims=False, multi_out=False):
    return elem_red_output_hrz(data0, data1, axis, keepdims, multi_out, poly_sch=True)

""" diamond fusion """
@akg.schedule(topi.cuda.reduce_opt.schedule_reduce)
def elem_red_diamond_manual(data0, data1, axis=None, keepdims=False, multi_out=False):
    return elem_red_diamond(data0, data1, axis, keepdims, multi_out, poly_sch=False)

def elem_red_diamond_auto(data0, data1, axis=None, keepdims=False, multi_out=False):
    return elem_red_diamond(data0, data1, axis, keepdims, multi_out, poly_sch=True)