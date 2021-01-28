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

"""
################################################

Testcase_PrepareCondition:

Testcase_TestSteps:

Testcase_ExpectedResult:


"""
import os
import pytest
from base import TestBase
from nose.plugins.attrib import attr
from test_run.detection_four2five_run import detection_four2five_run

############################################################
# TestCase= class: put to tests/*/
############################################################


class TestCase(TestBase):

    def setup(self):
        case_name = "test_akg_four2five"
        case_path = os.getcwd()
        self.params_init(case_name, case_path)
        self.caseresult = True
        self._log.info("============= {0} Setup case============".format(self.casename))
        self.testarg = [
            # testflag,opfuncname,testRunArgs, dimArgs
            # the shape is a tuple of (batch_size, box_number, width and lenth
            ("detection_four2five_001", detection_four2five_run, ((16, 0), "float16")),
            ("detection_four2five_002", detection_four2five_run, ((16, 1), "float16")),
            ("detection_four2five_003", detection_four2five_run, ((16, 2), "float16")),
            ("detection_four2five_004", detection_four2five_run, ((16, 3), "float16")),
            ("detection_four2five_005", detection_four2five_run, ((16, 4), "float16")),
            ("detection_four2five_006", detection_four2five_run, ((16, 5), "float16")),
        ]
        self.testarg_cloud = [
            # testflag,opfuncname,testRunArgs, dimArgs
        ]

        self.testarg_rpc_cloud = [
        ]

        return

    def test_run(self):
        self.common_run(self.testarg)

    def test_run_cloud(self):
        self.common_run(self.testarg_cloud)

    def test_run_rpc_cloud(self):
        self.common_run(self.testarg_rpc_cloud)

    def teardown(self):

        self._log.info("============= {0} Teardown============".format(self.casename))
        return
