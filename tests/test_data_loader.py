import pytest
from src.data.data_loader import DataLoader
from src.utils.helpers import define_system_parameters

@pytest.fixture
def data_loader():
    params = define_system_parameters()
    return DataLoader(params)

def test_generate_synthetic_data(data_loader):
    data = data_loader.generate_synthetic_data()
    assert 'X' in data
    assert 'Y' in data
    assert 'T' in data
    assert 'intensity' in data
    assert 'measurement_mask' in data
    assert 'measurement_indices' in data
    assert 'y_measurements' in data

def test_get_data_synthetic(data_loader):
    data = data_loader.get_data(use_synthetic=True)
    assert 'intensity' in data

def test_get_data_experimental(data_loader):
    with pytest.raises(NotImplementedError):
        data_loader.get_data(use_synthetic=False, file_path="dummy_path")