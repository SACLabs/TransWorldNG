from pathlib import Path
import pandas as pd
import numpy as np
import pandas as pd
import os

path_cwd = '/mnt/workspace/wangding/Desktop/TransWorld/transworld'
os.chdir(Path(path_cwd).absolute())
print(os.getcwd())

exp_dir = Path(path_cwd).resolve().parent / "HighD"
scenario = 'highway02'
data_dir = exp_dir/scenario/'data'

start_step = 0
end_step = 1000

df = pd.read_csv(exp_dir/ scenario / '02_tracks.csv')
df['type'] = 'veh'
print('Load data done!')

df['id'] = ['veh'+str(i) for i in df['id']]
df['precedingId'] = ['veh'+str(i) for i in df['precedingId']]
df['followingId'] = ['veh'+str(i) for i in df['followingId']]


selected_columns = ['frame', 'id','type']
node_df = df[selected_columns]

new_column_names = ['step', 'name', 'type']
node_df = node_df.rename(columns=dict(zip(selected_columns, new_column_names)))


######################### Get Node ##################################################

# Define the list of lane IDs to include in the new DataFrame
lane_ids = [2, 3, 5, 6]

# Initialize an empty list to hold the rows of the new DataFrame
rows = []

# Loop over the unique step values in node_df and the lane IDs
for step in node_df.step.unique():
    for lane_id in lane_ids:
        # Create a new row with the desired values
        row = {'step': step, 'name': 'lane'+str(lane_id), 'type': 'road'}

        # Append the row to the list of rows
        rows.append(row)

# Create a new DataFrame from the list of rows
result_df = pd.DataFrame(rows)
node_df = pd.concat([node_df, result_df], axis=0, ignore_index=True)
# Print the resulting DataFrame

node_df.to_csv(data_dir/'node_all.csv', index=False)

node_df = node_df[node_df['step']>start_step]
node_df = node_df[node_df['step']<end_step]
node_df.to_csv(data_dir/'node.csv', index=False)

print('Node done!')
######################### Get Edge ##################################################
df = df[df['frame']>start_step]
df = df[df['frame']>end_step]
# Create an empty list to store the edges
edges = []

# Iterate through the rows of the data frame
for index, row in df.iterrows():
    
    # Check if the precedingId is not zero
    if row['precedingId'] != 'veh0':
        # Add an edge from the precedingId to the current id
        edges.append((row['frame'],row['precedingId'], row['id'], 'veh_phy/ahead_veh'))
    # Check if the followingId is not zero
    if row['followingId'] != 'veh0':
        # Add an edge from the current id to the followingId
        edges.append((row['frame'],row['id'], row['followingId'], 'veh_phy/ahead_veh'))
    
    edges.append((row['frame'],row['id'],'lane'+str( row['laneId']), 'veh_phy/to_lane'))

# Create a new data frame from the edges
new_df = pd.DataFrame(edges, columns=['step','from', 'to', 'relation'])
new_df = new_df.drop_duplicates()
# Print the new data frame

new_df.to_csv(data_dir/'edge.csv', index=False)


print('Edge done!')
######################### Get Feat Vehicle ##################################################

df_car1 = df[df["type"]=='veh']
# Select the desired columns
selected_columns = ['frame','id', 'x', 'y', 'width', 'height', 'xVelocity', 'yVelocity',
       'xAcceleration', 'yAcceleration']
node_veh1 = df_car1[selected_columns]

# Reset the column names
new_column_names = ['step', 'name', 'x', 'y', 'width', 'height', 'xVelocity', 'yVelocity',
       'xAcceleration', 'yAcceleration']
node_veh1 = node_veh1.rename(columns=dict(zip(selected_columns, new_column_names)))

# Print the new data frame
node_veh1.to_csv(data_dir/'feat_veh.csv', index=False)


print('Feat veh done!')
######################### Get Feat Lane ##################################################

# Define the list of lane IDs to include in the DataFrame
lane_ids = [2, 3, 5, 6]

# Define the list of step values to include in the DataFrame
steps = list(range(start_step, end_step))

# Initialize an empty list to hold the rows of the DataFrame
rows = []

# Loop over the step and lane ID combinations
for step in steps:
    for lane_id in lane_ids:
        # Generate random values for the features
        name = f"lane{lane_id}"
        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)
        width = np.random.uniform(1, 5)
        speed_avg = np.random.uniform(1, 30)
        

        # Create a new row with the generated values
        row = {'step': step, 'name': name, 'x': x, 'y': y, 'width': width,
               'speed_avg': speed_avg}

        # Append the row to the list of rows
        rows.append(row)

# Create a new DataFrame from the list of rows
lane_features_df = pd.DataFrame(rows)

# Print the resulting DataFrame
lane_features_df.to_csv(data_dir/'feat_lane.csv', index=False)


print('Feat lane done!')