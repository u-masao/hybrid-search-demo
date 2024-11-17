import warnings
import pytest

@pytest.fixture(autouse=True)
def suppress_pydantic_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
