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

import os
import pytest

from base import TestBase
from test_run.matrix_diag_part_run import matrix_diag_part_run


class TestCase(TestBase):
    def setup(self):
        """set test case """
        case_name = "test_matrix_diag_part_001"
        case_path = os.getcwd()

        # params init
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("matrix_diag_part_001", matrix_diag_part_run, ((2, 3), "uint8")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((3, 2, 2, 3), "int32")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((2, 3, 4, 2), "float16")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((4, 4, 2), "float32")),
        ]
        self.testarg_rpc_cloud = [
            # testflag,opfuncname,testRunArgs, setdimArgs
            ("matrix_diag_part_001", matrix_diag_part_run, ((2, 3), "uint8")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((3, 2), "int32")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((2, 3), "int8")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((3, 2), "int32")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((2, 3), "int32")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((3, 2, 3), "int32")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((2, 3, 2), "int32")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((4, 4), "float16")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((233, 3), "float16")),
            ("matrix_diag_part_001", matrix_diag_part_run, ((4, 4), "float32")),
        ]

    def test_run(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg)

    def test_run_rpc_cloud(self):
        """
        run case.#
        :return:
        """
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):
        """
        clean environment
        :return:
        """
        self._log.info("============= {0} Teardown============".format(self.casename))
