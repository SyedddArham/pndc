# Distributed Edge Detection

## Requirements
- Python 3.8+
- MPI implementation (OpenMPI recommended)
- Virtual environment

## Installation 
- python -m venv venv
- source venv/bin/activate
- pip install -r requirements.txt

## Configuration
Edit `config.yaml` to customize:
- Input/output directories
- Image sizes
- Operators
- Logging level

## Running
- mpirun -n <num_processors> python parallel_edge_detection.py --config config.yaml

## Monitoring
Check `edge_detection.log` for detailed execution logs.

## Results
Results and performance reports are saved in the output directory:
- Processed images
- Performance metrics
- Analysis plots
- HTML report