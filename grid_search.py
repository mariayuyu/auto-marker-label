import itertools
import os
import shutil
import automarkerlabel as aml
import datetime

# --------------------------- grid_search.py -----------------------
# Train using all possible combinations of hyperparameters 
# Save results in grid_results folder 
# -------------------------------------------------------------------

# Define the parameter grid
param_grid = {
    'batch_size': [32, 64, 100],
    'nLSTMcells': [128, 256],
    'LSTMdropout': [0.17],
    'lr': [0.005, 0.078],
    'momentum': [0.65],
}

# Create all combinations of parameters
keys, values = zip(*param_grid.items())
param_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]

# Define fixed training parameters
fld = os.path.join('.', 'data')
datapath = os.path.join(fld, 'train')
markersetpath = os.path.join(fld, 'markers85.xml')
fs = 200
num_epochs = 50
prevModel = None
tempCkpt = None
contFromTemp = False
windowSize = 120
alignMkR = 'RFTC'
alignMkL = 'LFTC'

# Directory to store all grid search results
os.makedirs('grid_results', exist_ok=True)

# Loop over all parameter combinations
for i, params in enumerate(param_combinations):
    print(f"\n--- Running configuration {i+1}/{len(param_combinations)} ---")
    print(params)

    # Update automarkerlabel's hyperparameters (if stored as global vars)
    aml.batch_size = params['batch_size']
    aml.nLSTMcells = params['nLSTMcells']
    aml.LSTMdropout = params['LSTMdropout']
    aml.lr = params['lr']
    aml.momentum = params['momentum']
    
    # Unique save path for this config
    run_dir = os.path.join('grid_results', f'run_{i+1}')
    os.makedirs(run_dir, exist_ok=True)

    # Temporary CSV path
    aml.csv_path = os.path.join(run_dir, 'metrics.csv')

    # Run training
    aml.trainAlgorithm(
        savepath=run_dir,
        datapath=datapath,
        markersetpath=markersetpath,
        fs=fs,
        num_epochs=num_epochs,
        prevModel=prevModel,
        windowSize=windowSize,
        alignMkR=alignMkR,
        alignMkL=alignMkL,
        tempCkpt=tempCkpt,
        contFromTemp=contFromTemp
    )
