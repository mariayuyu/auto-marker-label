import ezc3d
import numpy as np
import os
import trackpy.predict
import trackpy
import pandas
import btk

def remove_uname_markers(input_path, output_path=None):

    c3d = ezc3d.c3d(input_path)
    marker_names = c3d['parameters']['POINT']['LABELS']['value']

    # Identify indices to keep
    keep_indices = [i for i, name in enumerate(marker_names) if not name.lower().startswith("uname")]
    print(f"Keeping {len(keep_indices)} of {len(marker_names)} markers")

    # Filter marker data
    c3d['data']['points'] = c3d['data']['points'][:, keep_indices, :]

    # Clean up per-marker metadata
    for key in list(c3d['parameters']['POINT'].keys()):
        param = c3d['parameters']['POINT'][key]
        if isinstance(param, dict) and 'value' in param:
            val = param['value']
            # Trim arrays/lists matching original marker count
            if isinstance(val, (list, np.ndarray)) and len(val) == len(marker_names):
                c3d['parameters']['POINT'][key]['value'] = [val[i] for i in keep_indices] if isinstance(val, list) else val[keep_indices]
            # Remove mismatched marker arrays
            elif isinstance(val, (list, np.ndarray)) and len(val) != len(keep_indices) and len(val) != 1:
                print(f"Removing POINT/{key} due to size mismatch")
                del c3d['parameters']['POINT'][key]

    # Optional cleanup
    if 'meta_points' in c3d['data']:
        del c3d['data']['meta_points']

    # Save result
    if output_path is None:
        output_path = os.path.splitext(input_path)[0] + "_cleaned.c3d"

    c3d.write(output_path)
    print(f"Saved cleaned file to: {output_path}")
    
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
    
def denoise_particles(Positions, T_appearance, T_disappearance, duration_cutoff, distance_cutoff, max_abs_y):
    # Remove particles which are visible for duration_cutoff or less 
    keep_indices    = T_disappearance - T_appearance > duration_cutoff
    print('%i particles removed because they were too short'%(len(T_appearance)-len(keep_indices)))
    Positions       = Positions[keep_indices]
    T_appearance    = T_appearance[keep_indices]
    T_disappearance = T_disappearance[keep_indices]
    
    # Remove instants when a particle is too close to other particles
    # Recursively, starting with the bit of shortest duration, removes its overlaps with the other bits
    durations       = T_disappearance - T_appearance
    sort_indices    = np.argsort(durations)
    Positions       = Positions[sort_indices]
    T_appearance    = T_appearance[sort_indices]
    T_disappearance = T_disappearance[sort_indices]
    
    nb_particles = len(T_appearance)
    nb_timepoints_removed = 0
    for nb in range(nb_particles-1):
        t_appearance    = int(T_appearance[nb])
        t_disappearance = int(T_disappearance[nb])
        position        = Positions[nb,t_appearance:t_disappearance]
        other_positions = Positions[nb+1:,t_appearance:t_disappearance]
        
        # calculate the distance to each other bit
        difference = other_positions - np.array([position])
        distance   = np.sqrt(np.sum(difference**2, axis = 2)) 
        
        # at every time point, what is the distance to the nearest bit
        min_distance = np.min(distance, axis = 0) # time
        
        # if this distance is too small, remove that timepoint from the bit
        Positions[nb,t_appearance:t_disappearance][min_distance < distance_cutoff] = np.zeros((t_disappearance-t_appearance, 3))[min_distance < distance_cutoff]
        nb_timepoints_removed += np.sum(min_distance < distance_cutoff)
    print('%i timepoints removed because the particle was too close to other particles'%nb_timepoints_removed)
    
    
    # Remove particles which are outside the forceplates 
    NewPositions = []
    for position in Positions:
        if np.nanmax(np.abs(position[:,1])) < max_abs_y:
            NewPositions.append(position)
    print('%i particles removed because they were outside the forceplates'%(nb_particles-len(NewPositions)))
    
    return np.array(NewPositions)
    
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
    print("particles done")
    datasheet = pandas.DataFrame({'x':x, 'y':y, 'z':z, 'frame': f})
    
    ## Automatic tracking
    predictor = trackpy.predict.NearestVelocityPredict(span = span) # takes the speed of the markers into account in order to guess where they will appear in the next timeframe
    print("automatic tracking done")
    traj      = predictor.link_df(datasheet, distance, memory = memory, pos_columns = ['x','y','z'])
    print("automatic tracking done")
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
    print("transforming done")
    for nb in range(nb_particles):

        indices = (p == nb) # rows of the dataframe which correspond to that particle

        Time = t[indices] # time during which that particle is visible
        time = np.array(Time, dtype = int)
        
        Visibility[nb][time] = 1
        
        Positions[nb,:,0][time] = x[indices]
        Positions[nb,:,1][time] = y[indices]
        Positions[nb,:,2][time] = z[indices]
    
    return Positions, Visibility
    
def save_to_c3d(Positions, Visibility, filename, Frequency = 200):

    nb_particles, nb_frames, _ = Positions.shape

    # Filter out markers with no visibility at all
    valid = np.sum(Visibility, axis=1) > 0
    Positions = Positions[valid]
    Visibility = Visibility[valid]
    nb_particles = Positions.shape[0]

    acq = btk.btkAcquisition()
    acq.Init(nb_particles, nb_frames)
    acq.SetPointFrequency(Frequency)
    acq.Update()

    for nb, position in enumerate(Positions):
        # BTK expects a shape of (nb_frames, 3)
        if not np.all(np.isnan(position)):
            Point = btk.btkPoint(0)
            Point.SetValues(position)  # shape: (nb_frames, 3)
            visibility = Visibility[nb]
            residual = np.zeros(nb_frames)
            residual[visibility == 0] = -1
            Point.SetResiduals(residual)
            Point.SetLabel(f'unlabelled_{nb:03d}')
            Point.Update()
            acq.AppendPoint(Point)

    acq.Update()

    writer = btk.btkAcquisitionFileWriter()
    writer.SetFilename(filename)
    writer.SetInput(acq)

    try:
        writer.Update()
        print(f"Successfully saved: {filename}")
    except Exception as e:
        print("Error during save:", e)
        import traceback
        traceback.print_exc()

# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":

    # Parameters for denoising the particles extracted by Qualysis
    duration_cutoff = 5    # number of samples, if a marker is visible for duration_cutoff samples or less, it is removed
    distance_cutoff = 5    # mm, if a marker is within distance_cutoff mm of another marker, it is removed
    max_abs_y       = 1000 # mm, if the absolute y position of a marker is larger than this, it is removed

    # Tracking parameters
    memory   = 100 # number of frames for which a marker is kept in memory after it disappears
    distance = 5   # maximum distance that a marker can travel in 1 timeframe (increase for walking)
    span     = 10  # Compute velocity from the most recent span+1 frames

    subject = 'Subj_60_2'
    input_file = '2-limb_02_1'
    input_path = f"data/{subject}/{input_file}.c3d"

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
    print("Sample Y values:", [np.nanmax(np.abs(p[:,1])) for p in Positions if not np.all(np.isnan(p[:,1]))])


    print('Denoising particles')
    denoised_Positions = denoise_particles(Positions, T_appearance, T_disappearance, duration_cutoff, distance_cutoff, max_abs_y)
    print("Positions after denoising:", denoised_Positions.shape)
    print('Tracking')
    tracked_Positions, Visibility = tracking(denoised_Positions, memory, distance, span)
    print("Positions after tracking:", tracked_Positions.shape)
    print('Saving to new c3d')
    output_path = f"data/{subject}/{input_file}_tracked.c3d"
    save_to_c3d(tracked_Positions, Visibility, output_path)

    remove_uname_markers(output_path)
