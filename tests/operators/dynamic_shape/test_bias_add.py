import boot
import pytest

def test_bias_add():
    boot.run("test_resnet50_bias_add_000", "bias_add_run", ([32, 10], "DefaultFormat", "float32"), "dynamic")
    boot.run("test_resnet50_bias_add_001", "bias_add_run", ([32, 1001], "DefaultFormat", "float32"), "dynamic")
    boot.run("test_resnet50_bias_add_002", "bias_add_run", ([32, 10], "DefaultFormat", "float16"), "dynamic")
    boot.run("test_resnet50_bias_add_003", "bias_add_run", ([32, 1001], "DefaultFormat", "float16"), "dynamic")
    
