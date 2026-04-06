# MDM2_Project2

## scripts/generate_example_data.py

generates timeseries of simulated particles in a 2D flow
each line is a subsequence frame
lines contain 2D coordinates like: x0,y0,x1,y1,...xN,yN

## scripts/visualize.py

convert example data into a visualization using matplotlib animation

## scripts/method1.py

in-progress implementation of method1 for velocity field extraction

# Notes

## Usage

### Configure python environment

1. Run `python -m venv .venv` to create a new virtual environment.
2. Run `source venv/bin/activate` to enter the virtual environment (this is different on Windows).
3. Upgrade venv's package manager installation (`pip install --upgrade pip`).
4. Install Python requirements (`pip install -r requirements.txt`).

### Generate example data

1. Run `scripts/generate_example_data.py --count 50 --frames 60 --extent 6 --seed 12345 --dissipation 0.5 > data.csv`.
   See `scripts/generate_example_data.py --help` for information on this command.
2. An animation of the generated data can be created using `scripts/visualize.py --extent 6 --fps 20 --output example.gif < data.csv`.

### Process generated data

Run `scripts/method1.py --extent 6.5 < data.csv` to process the generated data.
This extracts coherent velocity information from the data and writes visualizations to a `figures/` directory.
