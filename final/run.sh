#!/bin/sh
set -e

# PARAMETERS

COUNT=1000 # number of particles (data generation)
TIMESTEP=0.1 # simulated timestep of each frame (data generation)
DISSIPATION=0.1 # velocity dissipation constant (data generation)
EPSILON=1.0 # initial assignment affinity parameter (methods)

# PYTHON SETUP

if [ ! -d ".venv" ]; then
    echo "Setting up virtual environment"

    python -m venv .venv
    pip install --upgrade pip
    pip install -r requirements.txt
else
    echo "Skipped setting up virtual environment (already exists)"
fi

echo "Entering virtual environment"
source .venv/bin/activate

# DATA GENERATION
# Initial example data generation

echo "Generate example data"
python generate_example_data.py --count $COUNT --frames 60 --extent 6 --seed 12345 --timestep $TIMESTEP --dissipation $DISSIPATION > data.csv

echo "Generating visualization"
python visualize_example_data.py --extent 6.5 --fps 20 --output visualization.gif < data.csv &

echo "Generating PIV source images"
python generate_piv_images.py --size 256 --extent 6.1 --blobsize 2 --outdir piv_images < data.csv

# VELOCITY ESTIMATION SCRIPTS
# These convert input data (or images, for PIV) into an identical output format which can be evaluated

echo "Calculating PIV velocity estimates"
python run_piv.py --indir piv_images --outdir piv_renders --outfile piv_velocities.csv --count 60 &

echo "Calculating Delaunay velocity estimates"
python run_delaunay.py --infile data.csv --outfile delaunay_velocities.csv --frames-dir delaunay_renders --epsilon $EPSILON --extent 6 --sample-count 50 --edgepoints 30 &

echo "Calculating Voronoi velocity estimates"
python run_voronoi.py --infile data.csv --outfile voronoi_velocities.csv --frames-dir voronoi_renders --epsilon $EPSILON --extent 6 --sample-count 50 --edgepoints 30 &

echo "Calculating Radial Basis Function velocity estimates"
python run_rbf.py --infile data.csv --outfile rbf_velocities.csv --epsilon $EPSILON --extent 6 --sample-count 50 --edgepoints 30 &

wait # wait for each velocity estimation script to finish

# EVALUATION SCRIPTS
# These generate csv files describing the calculated MSE error of predicted velocities each frame

echo "Evaluating PIV velocity estimates"
python evaluate_results.py --infile piv_velocities.csv --outfile piv_errors.csv --timestep $TIMESTEP --dissipation $DISSIPATION &

echo "Evaluating Delaunay velocity estimates"
python evaluate_results.py --infile delaunay_velocities.csv --outfile delaunay_errors.csv --timestep $TIMESTEP --dissipation $DISSIPATION &

echo "Evaluating Voronoi velocity estimates"
python evaluate_results.py --infile voronoi_velocities.csv --outfile voronoi_errors.csv --timestep $TIMESTEP --dissipation $DISSIPATION &

echo "Evaluating Radial Basis Function velocity estimates"
python evaluate_results.py --infile rbf_velocities.csv --outfile rbf_errors.csv --timestep $TIMESTEP --dissipation $DISSIPATION &

wait # wait for evaluations to finish
