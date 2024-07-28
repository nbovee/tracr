import pytest
import sys
from unittest.mock import MagicMock
from pathlib import Path

# Add the project root directory to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# Mock for BaseDataset
class MockBaseDataset(MagicMock):
    DATA_SOURCE_DIRECTORY = Path('/mock/data/source/directory')

# Mock for torch
class MockDataset(MagicMock):
    pass

class MockUtils:
    class data:
        Dataset = MockDataset

class MockTorch:
    utils = MockUtils()

mock_torch = MockTorch()

# Mock for PIL
class MockPIL:
    Image = MagicMock()

mock_pil = MockPIL()

# Mock for torchvision
class MockTransforms:
    Compose = MagicMock()
    Resize = MagicMock()
    CenterCrop = MagicMock()
    ToTensor = MagicMock()
    Normalize = MagicMock()

class MockTorchvision:
    transforms = MockTransforms()

mock_torchvision = MockTorchvision()

def pytest_sessionstart(session):
    sys.modules['torch'] = mock_torch
    sys.modules['torch.utils'] = mock_torch.utils
    sys.modules['torch.utils.data'] = mock_torch.utils.data
    sys.modules['PIL'] = mock_pil
    sys.modules['torchvision'] = mock_torchvision
    sys.modules['torchvision.transforms'] = mock_torchvision.transforms
    
    # Mock BaseDataset
    from src.tracr.experiment_design.datasets.dataset import BaseDataset
    BaseDataset.__bases__ = (MockBaseDataset,)

@pytest.fixture(autouse=True)
def mock_imports():
    original_modules = {}
    for module in ['torch', 'torch.utils', 'torch.utils.data', 'PIL', 'torchvision', 'torchvision.transforms']:
        if module in sys.modules:
            original_modules[module] = sys.modules[module]
    
    sys.modules['torch'] = mock_torch
    sys.modules['torch.utils'] = mock_torch.utils
    sys.modules['torch.utils.data'] = mock_torch.utils.data
    sys.modules['PIL'] = mock_pil
    sys.modules['torchvision'] = mock_torchvision
    sys.modules['torchvision.transforms'] = mock_torchvision.transforms
    
    yield
    
    for module, original in original_modules.items():
        sys.modules[module] = original