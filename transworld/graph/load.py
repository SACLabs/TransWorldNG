import pandas as pd
from pathlib import Path
from graph.process import (
    generate_feat_dict,
    generate_graph_dict,
    generate_unique_node_id,
)
from typing import Dict, Tuple
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler

def check_file_exist(file_path: Path):
    if not Path(file_path).exists():
        raise FileNotFoundError("File path does not exist")

def unique_id(data_path: Path):
    check_file_exist(data_path)
    node_all = pd.read_csv(data_path / "node_all.csv")
    node_id_dict = generate_unique_node_id(node_all)
    return node_id_dict

def load_graph(data_path: Path, start_step, end_step,node_id_dict) -> Tuple[Dict, Dict]:
    """
    Load graph data from csv files.
    - read node,edge,feature files, assign unique node id to each node
    - generate graph dict, feature dict
    return: dicts containing the graph data, feature data
    """

    check_file_exist(data_path)
    all_files = list(data_path.glob("*.csv"))
    #node_all = pd.read_csv(data_path / "node_all.csv")
    #node_id_dict = unique_id(node_all)


    """Read files and get unique node id"""
    dfs: Dict = defaultdict(dict)
    file_names = ["node", "edge", "feat_veh", "feat_lane", "feat_road", "feat_tlc"]
    feat_file_names = ["feat_veh", "feat_lane", "feat_road", "feat_tlc"]
    #except_columns = ['name', 'coordinate',"shape"]
    scalers = {'veh': MinMaxScaler(), 'lane': MinMaxScaler(), 'road': MinMaxScaler(), 'tlc': MinMaxScaler()}
    
    for data_file in all_files:
        df_name = data_file.stem
        df = pd.read_csv(data_file)
        
        if df_name in feat_file_names:
            #df = df.loc[(df['step'] >= start_step) & (df['step'] <= end_step)]
            if df_name == 'feat_veh':
                df['coor_x'] = [float(coor.replace('(','').replace(")", '').split(',')[0]) for coor in df['coordinate']]
                df['coor_y'] = [float(coor.replace('(','').replace(")", '').split(',')[1]) for coor in df['coordinate']]
                select_columns=df.columns.difference(['step', 'name','coordinate'])
                scaled = pd.DataFrame(scalers['veh'].fit_transform(df[select_columns]), columns=select_columns, index=df.index)
                for col in select_columns:
                    df[col] = scaled[col]
                df = df.drop(['coordinate'], axis=1)
            elif df_name == 'feat_lane':
                df['shape_a'] = [float(coor.replace('(','').replace(")", '').split(',')[0]) for coor in df['shape']]
                df['shape_b'] = [float(coor.replace('(','').replace(")", '').split(',')[1]) for coor in df['shape']]
                df['shape_c'] = [float(coor.replace('(','').replace(")", '').split(',')[2]) for coor in df['shape']]
                df['shape_d'] = [float(coor.replace('(','').replace(")", '').split(',')[3]) for coor in df['shape']]
                select_columns=df.columns.difference(['step', 'name','shape'])
                scaled = pd.DataFrame(scalers['lane'].fit_transform(df[select_columns]), columns=select_columns, index=df.index)
                for col in select_columns:
                    df[col] = scaled[col]
                df = df.drop(['shape'], axis=1)
            elif df_name == 'feat_road':
                select_columns=df.columns.difference(['step', 'name'])
                scaled = pd.DataFrame(scalers['road'].fit_transform(df[select_columns]), columns=select_columns, index=df.index)
                for col in select_columns:
                    df[col] = scaled[col]
            elif df_name == 'feat_tlc':
                select_columns=df.columns.difference(['step', 'name'])
                scaled = pd.DataFrame(scalers['tlc'].fit_transform(df[select_columns]), columns=select_columns, index=df.index)
                for col in select_columns:
                    df[col] = scaled[col]        
            else:
                pass
        
        dfs[df_name] = df
        if df_name == "edge":
            dfs[df_name]["to_id"] = [node_id_dict[i] for i in dfs[df_name]["to"]]
            dfs[df_name]["from_id"] = [node_id_dict[i] for i in dfs[df_name]["from"]]
        elif df_name in file_names:
            dfs[df_name]["node_id"] = [
                node_id_dict[str(i)] for i in dfs[df_name]["name"]
            ]
        else:
            pass

    """Generate graph dict"""
    graph_dict = generate_graph_dict(dfs["edge"], start_step, end_step)
    #print(graph_dict)
    """Generate feature dict"""
    feat_dict = generate_feat_dict("lane", dfs["feat_lane"], start_step, end_step)
    feat_dict.update(generate_feat_dict("veh", dfs["feat_veh"], start_step, end_step))
    feat_dict.update(generate_feat_dict("road", dfs["feat_road"], start_step, end_step))
    feat_dict.update(generate_feat_dict("tlc", dfs["feat_tlc"], start_step, end_step))
    #print(feat_dict)
    return graph_dict, feat_dict, node_id_dict, scalers
