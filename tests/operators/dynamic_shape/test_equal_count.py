import boot
import pytest

def test_equal_count():
    boot.run("test_resnet50_equal_count_001", "equal_count_run", (((32,), (32,)), "int32", "equal_count"), "dynamic")
    