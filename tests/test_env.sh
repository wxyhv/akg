#!/bin/bash

# Copyright 2019 Huawei Technologies Co., Ltd
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
    echo "Usage: . ./test_env.sh [gpu|gpu-ci]"
}

CUR_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
AKG_DIR="${CUR_DIR}/.."
AD_BUILD_DIR="${AKG_DIR}/build"
TVM_ROOT="${AKG_DIR}/third_party/incubator-tvm"

export LD_LIBRARY_PATH=${AD_BUILD_DIR}:${LD_LIBRARY_PATH}
export PYTHONPATH=${TVM_ROOT}/python:${TVM_ROOT}/topi:${TVM_ROOT}/topi/python:${AKG_DIR}/tests/common:${AKG_DIR}/python:${AKG_DIR}/tests/fuzz/tune:${PYTHONPATH}

if [ $# -eq 1 ]; then
    case "$1" in
        "gpu")
            export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:${LD_LIBRARY_PATH}
            ;;
        "gpu_ci")
            pip_show_str=`pip show akg`
            str1=${pip_show_str#*Location:}
            str2=${str1%Requires*}
            str3=${str2%?}
            AKG_ROOT=${str3#* }
            TVM_ROOT="${AKG_ROOT}/third_party/incubator-tvm"
            LIBAKG_ROOT="${AKG_ROOT}/build"
            export LD_LIBRARY_PATH=${LIBAKG_ROOT}:${LD_LIBRARY_PATH}
            export PYTHONPATH=${TVM_ROOT}/python:${TVM_ROOT}/topi:${TVM_ROOT}/topi/python:${AKG_ROOT}/tests/common:${AKG_ROOT}/python:${AKG_ROOT}/tests/fuzz/tune:${PYTHONPATH}
            ;;
        *)
            usage
            ;;
    esac
else
    usage
fi

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
