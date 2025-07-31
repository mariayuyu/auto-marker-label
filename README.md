# Automatically Labeling MoCap 
This project is used to train a labeling model, clean up noisy raw C3D file and label it with the standard 85 marker layout. 
Implementation of: 

A. L. Clouthier, G. B. Ross, M. P. Mavor, I. Coll, A. Boyle and R. B. Graham, "Development and Validation of a Deep Learning Algorithm and Open-Source Platform for the Automatic Labelling of Motion Capture Markers," in IEEE Access, doi: 10.1109/ACCESS.2021.3062748.



## Virtual Environment & Packages
In the main folder you will find a YAML file: `label_env.yml` to create a virtual environment with all the necessary packages. 

Run this command in the Anaconda Prompt:
```
conda env create -f label_env.yml
```

You can now activate the environment in order to run the corresponding scripts.
```conda activate label_env```

If you do not wish to use the YAML file, here are the required steps and packages to start using a conda virtual environment:
```
conda create --name label_env

conda activate label_env

conda install conda-forge::trackpy

conda install pip

pip install _______
```

Packages for **preprocessing, labeling, training**:
* dash==2.18.2
* dash-bootstrap-components==1.7.1
* dash-core-components==2.0.0
* dash-html-components==2.0.0
* ezc3d==1.5.18
* h5py==3.13.0
* ipywidgets==8.1.5
* mydcc==0.1.22
* numpy==2.2.2
* pandas==2.2.3
* plotly==5.24.1
* scikit-learn==1.6.1
* scipy==1.15.1
* torch==2.7.0
* trackpy==0.6.4 (conda-forge)


## Preprocessing 
**`cleanup_raw_c3d.py`**: This file is used to disconnect particles, denoise particles, and track them in order to end up with a clean C3D file. 

In order to run the script, change the name of the `subject` folder (e.g. Subj_50), and `input_file` (e.g. 2-limb_02_1). 
You can also modify the parameters for denoising and tracking. In particular, `distance` parameter should be modified according to the action performed by the participant. If the number of resulting markers exceeds 255, `distance` will be tuned automatically. 

## Labeling 

**Several large files (the trained model) cannot be uploaded on GitHub** 

**[https://keeper.mpdl.mpg.de/d/c85ed3975bd64ac0864b/]**

**Replace the `data` folder with `data` folder downloaded through the link above**

**`markerLabelGUI.py`**, **`automarkerlabel.py`**: No modification or tuning is required. 

Follow the steps below to get started! 

### Setting up the GUI
* Set the paths to trained model, trained values pickle file, and market set in markerLabelGUI.py.
* Run markerLabelGUI.py, this will open the GUI in your browser.

### Using the GUI
* Enter the folder path where the c3d files to be labelled are located.
* Select the desired file from the dropdown menu.
* Click Load Data, the loading is complete when the plot of markers appears.
* If the person is not facing the +x direction, enter the angle to rotate the data about the z-axis (vertical) and click Submit. This angle should be chosen such that the person faces +x for the majority of the trial.
* Click Label Markers.
* Examine the results.
  * Clicking and dragging rotates, the scroll wheel zooms, and right clicking translates the plot. Hovering over a marker displays the marker number, label, and coordinates.
  * The slider at the bottom can be used to view different time frames. After clicking the slider, the left and right arrow keys can be used to adjust the timeframe as well.
  * The type of marker visualization can be selected from the Visualization dropdown menu. Confidence colours the markers based on the confidence in the predicted label. Unlabelled highlights unlabelled markers in red. Segments displays markers attached to the same body segment in the same colour. Skeleton displays a predefined skeleton for the 85 markerset.
  * The Error Detection box lists unlabelled markers and duplicated labels. The time frames where the error was detected are displayed. Note that the listed marker is guaranteed to be visible for the first and last frames, but may be missing from the intermediate frames of the listed range.
* Correct incorrect labels using the Modify Labels and Split Trajectories section. Type the number for the marker to change the label and type in its label name then click the Rename button. To leave a marker unlabelled, don't type anything and click Rename (this can be cleared by clicking the 'x').
* If a trajectory switches between two markers, it can be split in the Modify Labels and Split Trajectories section. Enter the marker number and the frame you wish to split it at and click Split. A new marker will be added containing the trajectory of that marker from the split frame onwards.
* Export a labelled version of the .c3d file by clicking the Export to C3D button. This will rotate the data back to the original orientation and fill small gaps through cubic spline interpolation. Unlablled markers will be exported as well with empty labels.
* Before loading a new c3d file, click the Refresh Settings button.

## Bugs
You may encounter several challenges during this process, please let me know if you do! But here are some bugs that I've encountered myself: 
* Can't label files with less than 85 trajectories.
* Pressing label button starts the process but in the end nothing happens - Refresh settings and restart, should see "finding labels and confidence scores" in the terminal.
* Preprocessed file is tiny (e.g. 3kb) - Change the `distance` parameter in `clean_raw_c3d.py` (5 for stationary, 16 for walking should be good but feel free to experiment). 
  



## Training
### Generate Simulated Trajectories
If you do not have existing labelled motion capture data to train the algorithm, simulated trajectories can be generated. 

If you have your own training data, skip this step.
- Download the body segment kinematics used to generate the simulated trajectories here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4293999.svg)](https://doi.org/10.5281/zenodo.4293999). Extract the .hdf5 file from the zip file and place it in the data folder.
- Next, the marker set to be used must be defined using an OpenSim marker set .xml file. 
  - Install [OpenSim](https://simtk.org/frs/index.php?group_id=91)
  - In OpenSim, Open Model Rajagopal2015_mod.osim included in /data folder
  - Right click on Markers in the Navigator to add new marker. Marker can be selected and moved in the visualization. Set the marker's name and parent_frame (ie body segment it is attached to) in the Properties window.
  - Save the marker set by right clicking Markers in the Navigator and choosing Save to File.
- Set parameters in **generateSimTrajectories.py**
  - Set the path to the marker set .xml file. 
  - Set the markers used to align the data. Ideally, these will be a marker on the left and right side of the torso or head (eg. right and left acromions). These are used to rotate the data about the verical axis so that the person faces the +x direction.
- Run **generateSimTrajectories.py**

### Train the Algorithm
- Set parameters in **trainAlgorithm.py**.
  - If using simulated trajectories to train, set the path to the marker set and to the pickle file generated by generateSimTrajectories.py.
  - If using existing labelled data to train, set the path to the folder containing the c3d files. Set the markers used to align the data. Ideally, these will be a marker on the left and right side of the torso or head (eg. right and left acromions).
  - Set patience for early stopping in `automarker.py`, `train_nn` function
- Define an OpenSim marker set, if not done already (see above).
- Run **trainAlgorithm.py**. The training will be performed on a GPU, if one is available. 
Note that this may take a long time to run (ie. hours - days). Training time can be reduced by using less training data (set num_participants in generateSimTrajectories.py).

### Transfer Learning
- As data is collected, labelled, and corrected, it can be added to the training set through transfer learning to improve the algorithm
- In **transferLearning.py**, set the path for the trained model and training values to build on (.ckpt and .pickle file) and the folder containing the .c3d files to add to the training set. Set the markers used to align the data. Ideally, these will be a marker on the left and right side of the torso or head (eg. right and left acromions).
- Run **transferLearning.py**
