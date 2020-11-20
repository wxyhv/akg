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

"""__init__"""
from .elem import elem_manual, elem_auto, elem_input_hrz_manual, elem_input_hrz_auto
from .elem import elem_output_hrz_manual, elem_output_hrz_auto, elem_diamond_manual, elem_diamond_auto
from .elem_red import elem_red_manual, elem_red_auto, elem_red_input_hrz_manual, elem_red_input_hrz_auto
from .elem_red import elem_red_output_hrz_manual, elem_red_output_hrz_auto, elem_red_diamond_manual, elem_red_diamond_auto
from .prim_pad import prim_pad_manual, prim_pad_auto, prim_pad_input_hrz_manual, prim_pad_input_hrz_auto
from .prim_pad import prim_pad_output_hrz_manual, prim_pad_output_hrz_auto, prim_pad_diamond_manual, prim_pad_diamond_auto
from .prim_unpad import prim_unpad_manual, prim_unpad_auto, prim_unpad_input_hrz_manual, prim_unpad_input_hrz_auto
from .prim_unpad import prim_unpad_output_hrz_manual, prim_unpad_output_hrz_auto, prim_unpad_diamond_manual, prim_unpad_diamond_auto
from .concat_prim import concat_prim_manual, concat_prim_auto
from .prim_transpose import prim_transpose_manual, prim_transpose_auto
from .prim_argmax import prim_argmax_manual, prim_argmax_auto, prim_argmax_input_hrz_manual, prim_argmax_input_hrz_auto
from .prim_argmax import prim_argmax_output_hrz_manual, prim_argmax_output_hrz_auto, prim_argmax_diamond_manual, prim_argmax_diamond_auto
from .prim_argmin import prim_argmin_manual, prim_argmin_auto, prim_argmin_input_hrz_manual, prim_argmin_input_hrz_auto
from .prim_argmin import prim_argmin_output_hrz_manual, prim_argmin_output_hrz_auto, prim_argmin_diamond_manual, prim_argmin_diamond_auto
from .special_elem_red import special_elem_red_manual, special_elem_red_auto, special_elem_red_input_hrz_manual, special_elem_red_input_hrz_auto
from .special_elem_red import special_elem_red_output_hrz_manual, special_elem_red_output_hrz_auto, special_elem_red_diamond_manual, special_elem_red_diamond_auto
