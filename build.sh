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

export AKG_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )/" && pwd )"
BUILD_DIR="${AKG_DIR}/build"
if [ -d "$BUILD_DIR" ]; then
    rm -rf $BUILD_DIR
fi
echo "mkdir $BUILD_DIR"
mkdir -p $BUILD_DIR

cd $BUILD_DIR

usage()
{
    echo "Usage:"
    echo "bash build.sh [-t gpu]"
    echo ""
    echo "Options:"
    echo "      -t hardware environment: gpu"
}

if [ ! -n "$1" ]; then
    echo "Must input paramter!"
    usage
    exit 1
fi

while getopts 't:' opt
do
    case "${opt}" in
        t)  
            if [ "${OPTARG}" == "gpu" ]; then
                cmake .. -DUSE_CUDA=ON -DUSE_RPC=ON
            else
                echo "Unkonwn parameter ${OPTARG}!"
                usage
                exit 1
            fi
            ;;
        *)
            echo "Unkonwn option ${opt}!"
            usage
            exit 1
    esac
done

if [ $? -ne 0 ]
then
    echo "[ERROR] CMake failed!!!"
    exit 1
fi

make -j16

if [ $? -ne 0 ]
then
    echo "[ERROR] make failed!!!"
    exit 1
fi

cd -
