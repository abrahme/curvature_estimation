import numpy as np
import pandas as pd
from typing import List, Tuple

def process_data_play(data: pd.DataFrame, partial_columns:List[str], target_columns: List[str],
                      position_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        data (pd.DataFrame): original data
        partial_columns (List[str]): column names describing partials of variables
        target_columns (List[str]): list of column names describing response columns 

    Returns:
        np.ndarray: numpy array number of partials x the length of data
    """

    manifold_dim = len(partial_columns)

    n_rows = data.shape[0]

    n_columns = (manifold_dim ** 2) - (manifold_dim - 1)

    design_matrix = np.zeros((n_rows, n_columns))
    i = 0
    for partial_a in partial_columns:
        for partial_b in partial_columns:
            if partial_a <= partial_b:
                design_matrix[:,i] = (data[partial_a] * data[partial_b]).to_numpy()
                i += 1
    
    full_design_matrix = np.vstack([design_matrix for _ in range(manifold_dim)])
    full_target = np.vstack([data[[target]].to_numpy() for target in target_columns])
    full_position = data[position_columns].to_numpy()
    full_dim_index = np.repeat(range(manifold_dim), n_rows)

    return full_design_matrix, full_target, full_position, full_dim_index


def process_full_data(data: pd.api.typing.DataFrameGroupBy, partial_columns:List[str], target_columns: List[str],
                      position_columns: List[str]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """_summary_

    Args:
        data (pd.api.typing.DataFrameGroupBy): grouped trajectories
        partial_columns (List[str]): partial columns
        target_columns (List[str]): target cols

    Returns:
        Tuple[np.ndarray]: tuple of predictors 
    """

    model_data = [process_data_play(play_df, partial_columns, target_columns, position_columns) for play_df in data]
    total_design_matrix = np.vstack([item[0] for item in model_data])
    total_target = np.vstack([item[1] for item in model_data])
    total_position = np.vstack([item[2] for item in model_data])
    total_dim_index = np.vstack([item[3] for item in model_data])

    return total_design_matrix, total_target, total_position, total_dim_index




