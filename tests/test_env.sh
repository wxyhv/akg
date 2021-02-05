#!/bin/bash

# Copyright 2019-2021 Huawei Technologies Co., Ltd
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

usage()
{
    echo "Usage: . ./test_env.sh [gpu]"
}

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MS_DIR="${CUR_DIR}/.."
AD_BUILD_DIR="${MS_DIR}/build"
TVM_ROOT="${AD_BUILD_DIR}/incubator-tvm"

export LD_LIBRARY_PATH=${AD_BUILD_DIR}:${LD_LIBRARY_PATH}
export PYTHONPATH=${TVM_ROOT}/python:${TVM_ROOT}/topi:${TVM_ROOT}/topi/python:${MS_DIR}/tests/common:${MS_DIR}/python:${MS_DIR}/tests/fuzz/tune:${PYTHONPATH}

if [ $# -eq 1 ]; then
    case "$1" in
        "gpu")
            echo "Configuration setting in gpu successfully."
            ;;
        *)
            echo "Configuration not set."
            usage
            ;;
    esac
else
    usage
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
