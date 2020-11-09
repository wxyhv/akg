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

from compute_791 import test_compute_791
from compute_1070 import test_compute_1070
from compute_1088 import test_compute_1088
from compute_1419 import test_compute_1419
from compute_1420 import test_compute_1420
from compute_1425 import test_compute_1425
from compute_1461 import test_compute_1461
from compute_1486 import test_compute_1486

def compute_791(poly_sch):
    test_compute_791((4096,), (4096, 768), (4096,), (4096, 768), (4096, 768), (4096,), (4096,),
    (768,), 'float32', poly_sch=poly_sch)

def compute_1070(poly_sch):
    test_compute_1070((32, 12, 128, 128), (32, 12, 128), (32, 12, 128, 128), (32, 12, 128, 128), 
    'float32', poly_sch=poly_sch)

def compute_1088(poly_sch):
    test_compute_1088((32, 2), (32, ), 'float32', poly_sch=poly_sch)

def compute_1419(poly_sch):
    test_compute_1419((4096, 3072), 'float32', poly_sch=poly_sch)

def compute_1420(poly_sch):
    test_compute_1420((4096, 768), (4096,), (4096, 768), (4096,), (4096,), (4096,),
    (32, 128, 768),  (768,), 'float32', poly_sch=poly_sch)

def compute_1425(poly_sch):
    test_compute_1425((640, 21128), (640,), (640,), (640, 21128), (640,), 'float32', poly_sch=poly_sch)

def compute_1461(poly_sch):
    test_compute_1461((32, 12, 128, 128), 'float32', poly_sch=poly_sch)

def compute_1486(poly_sch):
    test_compute_1486((32, 12, 128, 128), (32, 128), 'float32', 'int32', poly_sch=poly_sch)

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

    print("Available func:")
    print("\t", list(op_map.keys()), "\n")


if __name__ == '__main__':
    import sys
    import getopt
    import traceback
    from datetime import datetime

    op_map = {"791":compute_791, "1070":compute_1070, "1088":compute_1088, "1419":compute_1419, 
        "1420":compute_1420, "1425":compute_1425, "1461": compute_1461, "1486":compute_1486}

    all_f = list(op_map.values())
    op_map["all"] = all_f
    if len(sys.argv) == 1:
        usage(op_map)
        sys.exit()

    options, args = getopt.getopt(sys.argv[1:], "am", ["auto", "manual"])
    schedule_method = 'both'

    for name, value in options:
        if name in ("-a", "--auto"):
            schedule_method = "auto"
        if name in ("-m", "--manual"):
            schedule_method = "manual"

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
    for op in run_op_list:
        print("Operater: ", op.__name__)
        if schedule_method in ["both", "manual"]:
            try:
                print(" Time of manual schedule:")
                op(poly_sch=False)
            except:
                if op.__name__ in fail_op_list:
                    fail_op_list[op.__name__].extend(["using manual schedule:", traceback.format_exc()])
                else:
                    fail_op_list[op.__name__] = ["using manual schedule:", traceback.format_exc()]

        if schedule_method in ["both", "auto"]:
            try:
                print("Time of auto schedule:")
                op(poly_sch=True)
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
