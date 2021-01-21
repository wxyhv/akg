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

"""test composite json tuning"""
import sys
import time
import subprocess
from autotuning.job import launch_json

def run_json_tuning():
    time_start = time.time()
    if len(sys.argv) == 3:
        json_dir_ =  str(sys.argv[1])
        repo_path_ = str(sys.argv[2])
    else:
        json_dir_ =  "autotuning/shapes/bn_tune"
        repo_path_ = "../../../python/akg/composite/gpt3.json"
    debug_mode_ = True
    save_res_ = True
    all_space_ = False
    skip_exist_ = False
    extra_tune_ = False
    self_attrs = None
    tuning_attrs = ['enable_mark_multi_core',
                    'enable_pre_poly_loop_partition',
                    'enable_post_poly_loop_partition',
                    'enable_rewrite_scalar_compute',
                    'multicore_scalar_rearrange',
                    'pragma_remove_invariant_dependence',
                   ]
    launch_json(debug_mode=debug_mode_,
                save_res=save_res_,
                json_dir=json_dir_,
                repo_path=repo_path_,
                all_space=all_space_,
                skip_exist=skip_exist_,
                extra_tune=extra_tune_,
                self_attrs=self_attrs,
                tuning_attrs=tuning_attrs)
    time_end = time.time()
    print("launch time: ", time_end - time_start)

if __name__ == "__main__":
    if len(sys.argv) in (1, 3):
        run_json_tuning()
    else:
        print("please check args.")
