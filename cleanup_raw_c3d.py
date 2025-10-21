import ezc3d
import numpy as np
import os
import trackpy.predict
import trackpy
import pandas

# --------------------------------------------------------------------------- #
# --------------------------------- PARAMETERS ------------------------------ #
# --------------------------------------------------------------------------- #

# Parameters for denoising the particles extracted by Qualysis
duration_cutoff = 5    # number of samples, if a marker is visible for duration_cutoff samples or less, it is removed
distance_cutoff = 5    # mm, if a marker is within distance_cutoff mm of another marker, it is removed
max_abs_y       = 1000 # mm, if the absolute y position of a marker is larger than this, it is removed

# Tracking parameters
memory   = 100 # number of frames for which a marker is kept in memory after it disappears
distance = 5   # maximum distance that a marker can travel in 1 timeframe (5 for static, around 16 for walking)
span     = 10  # Compute velocity from the most recent span+1 frames

# Subject name and file name that you want preprocessed 
subject = 'Subj_55_1'
input_file = 'Gait_0002 - 6'
input_path = f"data/{subject}/{input_file}.c3d"

# --------------------------------------------------------------------------- #


print(f"input path: {input_path}")
print(f"running path: {os.getcwd()}")
print(f"path exists: {os.path.exists(input_path)}")

  
def load_c3d_markers(filepath):
    c3d = ezc3d.c3d(filepath)
    data = c3d['data']['points']  # shape: (4, N_MARKERS, N_FRAMES)

    # Get XYZ and reshape
    positions = data[:3, :, :]  # (3, n_markers, n_frames)
    traj_data = np.transpose(positions, (1, 0, 2))  # (n_markers, 3, n_frames)
    # Visibility: check for NaNs in XYZ positions
    visibility = ~np.isnan(traj_data).any(axis=1)  # shape: (n_markers, n_frames)

    # Print diagnostic info
    print("XYZ shape:", traj_data.shape)
    print("X min/max:", np.nanmin(traj_data[:, 0, :]), np.nanmax(traj_data[:, 0, :]))
    print("Y min/max:", np.nanmin(traj_data[:, 1, :]), np.nanmax(traj_data[:, 1, :]))
    print("Z min/max:", np.nanmin(traj_data[:, 2, :]), np.nanmax(traj_data[:, 2, :]))
    

    return traj_data, visibility.astype(int)

def disconnect_particles(Position, Visibility):
    nb_particles, nb_dim, nb_timepoints = np.shape(Position)
    mean_visibility = np.mean(Visibility, axis = 1)

    Fully_visible = []
    Partly_visible = []
    T_appearance   = []
    T_disappearance   = []

    for nb in range(nb_particles):
        
        ## Fully visible markers
        if mean_visibility[nb] == 1:		
            Fully_visible.append(Position[nb].T)

        ## Partly visible markers
        elif mean_visibility[nb] > 0 and mean_visibility[nb] < 1:
        
            # Whenever a particle disappears, the trajectories before and after the disappearance are split into separate particles
            V = Visibility[nb]
            while sum(V) > 0:
            
                ## Beginning of the first visible chunk
                t_appearance = np.argmax(V)
                
                ## End of the first visible chunk
                length = np.argmin(V[t_appearance:])
                # if the marker remains visible from t0 until the end of the trial:
                if length == 0:
                    t_disappearance = nb_timepoints 
                # if the marker becomes invisible before the end of the trial	
                else:
                    t_disappearance = t_appearance + length
                
                ## Add the chunk to Partly_visible
                particle = np.nan*np.ones((nb_timepoints,3))
                particle[t_appearance:t_disappearance] = Position[nb,:,t_appearance:t_disappearance].T
                Partly_visible.append(particle)
                T_appearance.append(t_appearance)
                T_disappearance.append(t_disappearance)
                
                ## Erase its visibility
                V[t_appearance:t_disappearance] = 0	

    nb_fully_visible  = len(Fully_visible)			
    nb_partly_visible = len(Partly_visible)
    nb_particles      = nb_fully_visible + nb_partly_visible
    Positions           = np.zeros((nb_particles,nb_timepoints,3))
    T_appearance_all    = np.zeros((nb_particles))
    T_disappearance_all = np.zeros((nb_particles))

    ## Fully visible markers 
    T_appearance_all[:nb_fully_visible] = 0
    T_disappearance_all[:nb_fully_visible] = nb_timepoints
    for nb in range(nb_fully_visible):
        Positions[nb] = Fully_visible[nb]

    ## Partly visible markers		
    for nb in range(nb_partly_visible):
        Positions[nb_fully_visible + nb]           = Partly_visible[nb]
        T_appearance_all[nb_fully_visible + nb]    = T_appearance[nb]
        T_disappearance_all[nb_fully_visible + nb] = T_disappearance[nb]
        
    return Positions, T_appearance_all, T_disappearance_all
    


def denoise_particles(Positions, T_appearance, T_disappearance,
                      duration_cutoff, distance_cutoff, max_abs_y,
                      max_dxy=1000, max_dz=1500, remove_short = False):
    if remove_short:
        # --- Remove particles that are too short ---
        durations = T_disappearance - T_appearance
        keep_indices = durations > duration_cutoff
        print(f'{np.sum(~keep_indices)} particles removed because they were too short')
        
        Positions = Positions[keep_indices]
        T_appearance = T_appearance[keep_indices]
        T_disappearance = T_disappearance[keep_indices]

    # --- Sort by duration (shortest first) ---
    durations = T_disappearance - T_appearance
    sort_indices = np.argsort(durations)
    Positions = Positions[sort_indices]
    T_appearance = T_appearance[sort_indices]
    T_disappearance = T_disappearance[sort_indices]

    # --- Remove points too close to others ---
    nb_particles = len(T_appearance)
    nb_timepoints_removed = 0
    for nb in range(nb_particles - 1):
        t_start = int(T_appearance[nb])
        t_end = int(T_disappearance[nb])
        pos = Positions[nb, t_start:t_end]
        others = Positions[nb+1:, t_start:t_end]

        diff = others - pos[None, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        min_dist = np.nanmin(dist, axis=0)

        mask = min_dist < distance_cutoff
        Positions[nb, t_start:t_end][mask] = 0
        nb_timepoints_removed += np.sum(mask)

    print(f'{nb_timepoints_removed} timepoints removed due to proximity to other particles')


    # --- When particles are outside the forceplate area, their position is set to nan ---
    outside_forceplates = np.abs(Positions[:,:,1]) > max_abs_y
    Positions[outside_forceplates] = np.nan
    print(f'{np.sum(np.any(outside_forceplates, axis = 1))} particles truncated because they were outside the forceplates')
    
    # --- Remove particles which are completely invisble ---
    non_nan_mask    = np.any(1 - np.isnan(Positions), axis=(1, 2))
    Positions       = Positions[non_nan_mask]
    
    print(f'{np.sum(1- non_nan_mask)} particles removed for being fully invisible')

    # --- Remove particles too far from center (axis-specific thresholds) ---
    center               = np.nanmean(Positions, axis = 0)
    difference_to_center = Positions - np.array([center])
    abs_mean_diffs       = np.abs(np.nanmean(difference_to_center, axis=1))
    keep_indices         = (abs_mean_diffs[:,0] < max_dxy)*(abs_mean_diffs[:,1] < max_dxy)*(abs_mean_diffs[:,2] < max_dz)
    Positions            = Positions[keep_indices]

    print(f"{np.sum(1-keep_indices)} particles removed for being too far from framewise center")


    return Positions

    
def tracking(Positions, memory, distance, span):

    nb_particles, duration, nb_dims = np.shape(Positions)
    times = np.arange(duration)

    ## Create panda DataFrame with all the markers	
    # Columns of the DataFrame: X, Y, Z, frame
    x = []
    y = []
    z = []
    f = []
        
    # Particles
    for position in Positions:
        invisible = np.any(np.isnan(position), axis = 1)
        
        ## Add all visible timepoints to the columns of the DataFrame
        x.extend(list(position[:,0][invisible == 0]))
        y.extend(list(position[:,1][invisible == 0]))
        z.extend(list(position[:,2][invisible == 0]))
        f.extend(list(times[invisible == 0]))	
    datasheet = pandas.DataFrame({'x':x, 'y':y, 'z':z, 'frame': f})
    
    ## Automatic tracking
    predictor = trackpy.predict.NearestVelocityPredict(span = span) # takes the speed of the markers into account in order to guess where they will appear in the next timeframe
    traj      = predictor.link_df(datasheet, distance, memory = memory, pos_columns = ['x','y','z'])
    ## Transform the output of the tracking into an array of size particles x time x dimension 
    x = np.array(traj['x'])
    y = np.array(traj['y'])
    z = np.array(traj['z'])
    t = np.array(traj['frame'])
    p = np.array(traj['particle'])

    nb_particles = int(max(p))
    duration = int(max(t)) + 1

    Positions  = np.nan*np.ones((nb_particles, duration, 3))
    Visibility = np.zeros((nb_particles, duration))
    for nb in range(nb_particles):

        indices = (p == nb) # rows of the dataframe which correspond to that particle

        Time = t[indices] # time during which that particle is visible
        time = np.array(Time, dtype = int)
        
        Visibility[nb][time] = 1
        
        Positions[nb,:,0][time] = x[indices]
        Positions[nb,:,1][time] = y[indices]
        Positions[nb,:,2][time] = z[indices]
    
    return Positions, Visibility

def save_to_c3d(Positions, Visibility, filename_out, Frequency=200):

    nb_markers, nb_frames, _ = Positions.shape

    # Filter out fully invisible markers
    valid = np.sum(Visibility, axis=1) > 0
    Positions = Positions[valid]
    Visibility = Visibility[valid]
    nb_markers = Positions.shape[0]

    # Transpose to shape (4, num_markers, num_frames)
    points = np.ones((4, nb_markers, nb_frames), dtype=np.float32) * np.nan
    points[0:3, :, :] = np.transpose(Positions, (2, 0, 1))  # x, y, z
    points[3, :, :] = 0.0  # residuals 0 for visible, -1 for invisible
    points[3, Visibility == 0] = -1.0

    # Create C3D structure
    c3d = ezc3d.c3d()
    c3d['data']['points'] = points

    # Add marker names
    labels = [f'unlabelled_{i:03d}' for i in range(nb_markers)]
    c3d['parameters']['POINT']['LABELS']['value'] = labels
    c3d['parameters']['POINT']['DESCRIPTIONS']['value'] = labels
    c3d['parameters']['POINT']['USED']['value'] = [nb_markers]
    c3d['parameters']['POINT']['FRAMES']['value'] = [nb_frames]
    c3d['parameters']['POINT']['RATE']['value'] = [Frequency]
    c3d['header']['points']['frame_rate'] = Frequency
    c3d['header']['points']['size'] = nb_markers

    # Save to file
    c3d.write(filename_out)
    print(f"Saved: {filename_out}")


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    print('Loading marker data from C3D')
    Position, Visibility = load_c3d_markers(input_path)
    print("Positions after loading:", Position.shape)
    print("Visibility sum per marker:", np.sum(Visibility, axis=1))
    print("Fully visible markers:", np.sum(np.mean(Visibility, axis=1) == 1))
    print("Partly visible markers:", np.sum((np.mean(Visibility, axis=1) > 0) & (np.mean(Visibility, axis=1) < 1)))

    print('Disconnecting particles')
    Positions, T_appearance, T_disappearance = disconnect_particles(Position, Visibility)
    print("Positions after disconnecting:", Positions.shape)
    print("Number of particles before forceplate check:", len(Positions))
    #print("Sample Y values:", [np.nanmax(np.abs(p[:,1])) for p in Positions if not np.all(np.isnan(p[:,1]))])


    print('Denoising particles')
    Positions = denoise_particles(Positions, T_appearance, T_disappearance, duration_cutoff, distance_cutoff, max_abs_y)
    
    print("Positions after denoising:", Positions.shape)
    
    print('Tracking')
    # Tuning distance parameter
    nb_markers = 1000
    while nb_markers > 255:   
        print("distance: ", distance)
        Positions, Visibility = tracking(Positions, memory, distance, span)
        print("Positions after tracking:", Positions.shape)
        nb_markers = Positions.shape[0]

        delta = nb_markers - 255
        distance += min(3, max(1, delta//175))

    print('Saving to new c3d')
    output_path = f"data/{subject}/{input_file}_tracked.c3d"
    save_to_c3d(Positions, Visibility, output_path)

