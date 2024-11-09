# LDV-Bessel Project

## Overview
This project provides tools and utilities for Laser Doppler Vibrometer (LDV) analysis using Bessel functions. It focuses on processing and analyzing vibration measurements to understand structural dynamics and material properties.

## Features
- Data acquisition from LDV systems
- Advanced signal processing with Bessel function analysis
- Frequency domain analysis capabilities
- Material properties characterization
- Visualization tools for displacement and velocity data

## Requirements
- Python 3.7+
- Required packages:
```
pip install -r requirements.txt
```

## Project Structure
- `src/`: Source code files
    - `models/`: Material and analysis models
    - `ldv/`: LDV interface and processing
- `docs/`: Documentation files
- `tests/`: Test files

## Quick Start
```python
from src.models.material import MaterialProperties
from src.ldv import LaserDopplerVibrometer

# Initialize material and LDV system
material = MaterialProperties.create_cardboard()
ldv = LaserDopplerVibrometer(system_params=system_params, material=material)

# Perform measurement
ldv.setup_measurement({
        "length": 0.1,
        "width": 0.1,
        "thickness": 0.001
})
results = ldv.measure_point(x=0.05, y=0.05, duration=1.0)
```

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing
Please read CONTRIBUTING.md for details on submitting pull requests.

## Contact
For questions or issues, please open an issue in this repository.