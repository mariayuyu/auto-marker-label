# Automatically Labeling MoCap 
This project is used to clean up noisy raw C3D file and label it with the standard 85 marker layout. 

**!!Several large files (the trained model) cannot be uploaded on GitHub, (https://drive.google.com/drive/folders/1DXdEskA8GN5VK8XUmYM03Nglp_H4wqA2?usp=drive_link)!!**

I recommend creating two different virtual environments for preprocessing and labeling as the packages may collide. 

## Preprocessing 
In the `Preprocessing` folder, you will find `cleanup_raw_c3d.py` file. This file is used to disconnect particles, denoise particles, and track them in order to end up with a clean C3D file. 

In order to run the script, change the name of the `subject` folder (e.g. Subj_50"), and `input_file` (e.g. 2-limb_02_1). 
You can also modify the parameters for denoising and tracking. In particular, `distance` parameter should be modified according to the action performed by the participant. (5 for static, 16 for gait). 

Several packages are necessary for this script, here's how to create a virtual environment if you are using Anaconda and install some packages:

conda create --name preprocess

conda activate preprocess

conda install python=3.11

conda install conda-forge::btk

conda install conda-forge::trackpy

Packages that are necessary:
* ezc3d==1.5.18
* numpy==1.26.4
* trackpy==0.6.4
* pandas==2.3.0
* btk==0.4.dev0
  
**Can only be run on Windows environment due to `btk` package**

## Labeling 
In the `Training` folder, you will find the trained model `best_model_0705.ckpt`, training values `training_stats_2025-07-05.pickle`, `markerLabelGUI.py` script.

Several packages are necessary: 
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
  * The type of marker visualization can be selected from the Visualization dropdown menu. Confidence colours the markers based on the confidence in the predicted label. Unlabelled highlights unlabelled markers in red. Segments displays markers attached to the same body segment in the same colour.
  * The Error Detection box lists unlabelled markers and duplicated labels. The time frames where the error was detected are displayed. Note that the listed marker is guaranteed to be visible for the first and last frames, but may be missing from the intermediate frames of the listed range.
* Correct incorrect labels using the Modify Labels and Split Trajectories section. Type the number for the marker to change the label and select the desired label from the dropdown then click the Rename button. To leave a marker unlabelled, leave the dropdown menu blank (this can be cleared by clicking the 'x').
* If a trajectory switches between two markers, it can be split in the Modify Labels and Split Trajectories section. Enter the marker number and the frame you wish to split it at and click Split. A new marker will be added containing the trajectory of that marker from the split frame onwards.
* Export a labelled version of the .c3d file by clicking the Export to C3D button. This will rotate the data back to the original orientation and fill small gaps through cubic spline interpolation. Unlablled markers will not be exported.
* Before loading a new c3d file, click the Refresh Settings button.

## Bugs
You may encounter several challenges during this process, please let me know if you do! But here are some bugs that I've encountered myself: 
* Can't label files with less than 85 markers - Currently working on this.
* Pressing label button starts the process but in the end nothing happens - Refresh settings and restart, should see "finding labels and confidence scores" in the terminal.
* Preprocessed file is tiny (e.g. 3kb) - Change the `distance` parameter in `clean_raw_c3d.py` (5 for stationary, 16 for walking should be good but feel free to experiment). 
  
