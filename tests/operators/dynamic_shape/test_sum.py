import boot
import pytest

def test_sum():
    boot.run("001_sum", "sum_run", ((1024, ), (0, ), False, "float32"),"dynamic")
    boot.run("001_sum", "sum_run", ((32, 1024 ), (1, ), False, "float32"),"dynamic")
    