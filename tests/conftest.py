import pytest
import torch


@pytest.fixture(autouse=True)
def fresh_compile_cache():
    # Each test builds new model shapes/configs; start below torch.compile's recompile limit.
    torch.compiler.reset()
