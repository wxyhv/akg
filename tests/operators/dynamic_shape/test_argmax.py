import boot
import pytest


def test_argmax():
    boot.run("test_resnet50_argmax_001", "argmax_run", ((32, 10), "float16", -1), "dynamic")
    boot.run("test_resnet50_argmax_002", "argmax_run", ((32, 10), "float32", -1), "dynamic")
    boot.run("test_resnet50_argmax_003", "argmax_run", ((32, 1001), "float16", -1), "dynamic")
    boot.run("test_resnet50_argmax_004", "argmax_run", ((32, 1001), "float32", -1), "dynamic")
    