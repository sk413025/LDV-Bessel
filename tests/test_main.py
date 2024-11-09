import pytest
import numpy as np
from src.models.material import MaterialProperties
from src.models.modal import ClassicalModalAnalysis, BesselModalAnalysis
from src.models.system import SystemParameters  # Fixed import path

@pytest.fixture
def material():
    return MaterialProperties.create_cardboard()

@pytest.fixture 
def system_params(material):
    return SystemParameters(material)  # Pass material as positional argument

@pytest.fixture
def box_dimensions():
    return {
        'length': 0.1,
        'width': 0.1, 
        'thickness': 0.001
    }

def test_classical_modal_init(system_params, box_dimensions):
    modal = ClassicalModalAnalysis(system_params, box_dimensions)
    assert modal.box_dimensions == box_dimensions
    assert modal.modal_frequencies == []
    assert modal.modal_shapes == []

def test_classical_modal_frequencies(system_params, box_dimensions):
    modal = ClassicalModalAnalysis(system_params, box_dimensions)
    frequencies = modal.calculate_modal_frequencies()
    assert len(frequencies) == 9  # 3x3 modes
    assert all(f > 0 for f in frequencies)  # All frequencies should be positive
    assert frequencies == sorted(frequencies)  # Should be sorted

def test_classical_modal_shapes(system_params, box_dimensions):
    modal = ClassicalModalAnalysis(system_params, box_dimensions)
    shapes = modal.calculate_modal_shapes()
    assert len(shapes) == 9  # 3x3 modes
    
    # Test shape function values
    x, y = 0.05, 0.05  # Test point
    for shape in shapes:
        val = shape(x, y)
        assert isinstance(val, (float, np.float64))
        assert -1 <= val <= 1  # Shape function should be normalized

def test_bessel_modal_init(system_params, box_dimensions):
    modal = BesselModalAnalysis(system_params, box_dimensions)
    assert modal.box_dimensions == box_dimensions
    assert modal.max_modes == (3, 3)
    assert len(modal.bessel_zeros) == 3

def test_bessel_modal_frequencies(system_params, box_dimensions):
    modal = BesselModalAnalysis(system_params, box_dimensions)
    frequencies = modal.calculate_modal_frequencies()
    assert len(frequencies) == 9  # 3x3 modes
    assert all(f > 0 for f in frequencies)
    assert frequencies == sorted(frequencies)

def test_bessel_modal_shapes(system_params, box_dimensions):
    modal = BesselModalAnalysis(system_params, box_dimensions)
    shapes = modal.calculate_modal_shapes()
    assert len(shapes) == 9
    
    # Test shape function values within and outside radius
    x, y = 0.02, 0.02  # Inside radius
    for shape in shapes:
        val = shape(x, y)
        assert isinstance(val, (float, np.float64))
        
    x, y = 0.1, 0.1  # Outside radius
    for shape in shapes:
        assert shape(x, y) == 0

def test_modal_response(system_params, box_dimensions):
    classical = ClassicalModalAnalysis(system_params, box_dimensions)
    bessel = BesselModalAnalysis(system_params, box_dimensions)
    
    # Calculate responses
    classical.calculate_modal_frequencies()
    classical.calculate_modal_shapes()
    bessel.calculate_modal_frequencies()
    bessel.calculate_modal_shapes()
    
    x, y, t = 0.05, 0.05, 0.001
    
    classical_response = classical.calculate_modal_response(x, y, t)
    bessel_response = bessel.calculate_modal_response(x, y, t)
    
    assert isinstance(classical_response, (float, np.float64))
    assert isinstance(bessel_response, (float, np.float64))