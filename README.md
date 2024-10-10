# LDV Intensity Field Reconstruction

This project implements a reconstruction method for Laser Doppler Vibrometer (LDV) intensity fields. It provides tools for simulating LDV measurements, reconstructing intensity fields, and visualizing results.

## Features

- Simulation of LDV measurements
- Reconstruction of intensity fields using optimization techniques
- Visualization of original and reconstructed intensity fields
- Analysis tools for reconstructed data

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/LDV-Bessel.git
   cd LDV-Bessel
   ```

2. Install the required packages:
   ```bash
   pip install -e .
   ```

3. Set up the PYTHONPATH:
   
   For Windows (PowerShell):
   ```powershell
   $env:PYTHONPATH = "$env:PYTHONPATH;C:\path\to\your\LDV-Bessel"
   ```
   
   For Linux/macOS:
   ```bash
   export PYTHONPATH=$PYTHONPATH:/path/to/your/LDV-Bessel
   ```

   Replace `/path/to/your/LDV-Bessel` with the actual path to your project directory.

## Usage

1. Configure the system parameters in `config/parameters.yaml`.

2. Run the main script:
   ```bash
   python main.py
   ```

3. For custom usage, import the necessary classes:
   ```python
   from src.models.intensity_reconstruction import IntensityReconstructor
   from src.visualization.visualizer import Visualizer
   from src.data.data_loader import DataLoader
   ```

## Running Tests

To run the tests, use the following command from the project root directory:

```bash
pytest tests/test_intensity_reconstruction.py
```

Make sure you have set up the PYTHONPATH as described in the Installation section before running the tests.

## Project Structure

- `src/`: Source code for the project
  - `models/`: Contains the IntensityReconstructor class
  - `visualization/`: Contains the Visualizer class
  - `data/`: Contains the DataLoader class
  - `utils/`: Utility functions
- `tests/`: Unit tests
- `config/`: Configuration files
- `main.py`: Main script to run the reconstruction

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
