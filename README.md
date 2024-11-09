# LDV-Bessel Project

## Overview
Tools for analyzing vibration measurements using both classical and Bessel function-based methods in Laser Doppler Vibrometry. Specializes in modal analysis and real-time signal processing.

## Features
- Dual modal analysis methods (Classical and Bessel)
- Real-time LDV signal processing
- Advanced frequency domain analysis
- Surface vibration modeling
- Interactive visualization tools
- Automated measurement sequences 

## Requirements
- Python 3.8+
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.4

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