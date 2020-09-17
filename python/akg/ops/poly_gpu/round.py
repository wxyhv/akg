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

"""round"""
from akg.ops.math_gpu.round import round
import akg.topi as topi
import akg

@akg.schedule(topi.cuda.schedule_injective)
def round_manual(x):
    """round"""
    return round(x)

def round_auto(x):
    """round"""
    return round(x)

