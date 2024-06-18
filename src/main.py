# Multi-Window Causal Discovery
#
# Author: Lukas Schild (lusc@hvl.no)

import numpy as np
import pandas as pd
import os
import math
import sklearn.preprocessing
from castle.algorithms import DAG_GNN
from castle.algorithms import NotearsNonlinear
from castle.algorithms import DirectLiNGAM
import torch
from tqdm import tqdm
from contextlib import contextmanager
import logging
import ruptures as rpt
from tigramite import data_processing as pp
from tigramite import plotting as tp
from tigramite.pcmci import PCMCI
from tigramite.lpcmci import LPCMCI
from tigramite.models import LinearMediation, Prediction
from tigramite.independence_tests.parcorr import ParCorr
import tomli


LINE_LENGTH = 56 # Length of lines in log output


@contextmanager
def all_logging_disabled(highest_level=logging.CRITICAL):
    """
    A context manager that will prevent any logging messages
    triggered during the body from being processed.
    :param highest_level: the maximum logging level in use.
      This would only need to be changed if a custom level greater than CRITICAL
      is defined.
    """
    # two kind-of hacks here:
    #    * can't get the highest logging level in effect => delegate to the user
    #    * can't get the current module-level override => use an undocumented
    #       (but non-private!) interface

    previous_level = logging.root.manager.disable

    logging.disable(highest_level)

    try:
        yield
    finally:
        logging.disable(previous_level)


# Print parameters
def print_params(params):
    half_size = math.floor(LINE_LENGTH/2)
    print('+'.ljust(LINE_LENGTH, '-') + '+')
    print('| Scaling type'.ljust(half_size, ' ') +f'| {params["scaling_type"]}'.ljust(half_size, ' ') + '|')
    print('| Use differences'.ljust(half_size, ' ') +f'| {params["use_differences"]}'.ljust(half_size, ' ') + '|')
    print('| Lags'.ljust(half_size, ' ') +f'| {params["lags"]}'.ljust(half_size, ' ') + '|')
    print('| Algorithm'.ljust(half_size, ' ') +f'| {params["causal_discovery_algo"]}'.ljust(half_size, ' ') + '|')
    print('| Target variable'.ljust(half_size, ' ') +f'| {params["target_variable"]}'.ljust(half_size, ' ') + '|')
    print('| Verbose'.ljust(half_size, ' ') +f'| {params["verbose"]}'.ljust(half_size, ' ') + '|')
    print('+'.ljust(LINE_LENGTH, '-') + '+')


# Print Info at start
def print_start():
    print()
    print('+'.ljust(LINE_LENGTH, '-') + '+')
    print('| Multi-Window Causal Discovery Algorithm (MWCD)'.ljust(LINE_LENGTH, ' ') + '|')
    print('+'.ljust(LINE_LENGTH, '-') + '+')
    print('|'.ljust(LINE_LENGTH, ' ') + '|')
    print('| Change Point Detection...'.ljust(LINE_LENGTH, ' ') + '|')
    print('|'.ljust(LINE_LENGTH, ' ') + '|')
    print('+'.ljust(LINE_LENGTH, '-') + '+')


def get_data_slice(df, start, end, params, reshape=True):
    '''
    Create a data slice for causal discovery algorithms
    from the gcastle package
    
    Args:
        df (DataFrame): Data
        start (str): Start date
        params (Namespace): Parameters

    Returns:
        X (np.array): Data slice
    '''

    slice_start = pd.Timestamp(start)
    slice_end = pd.Timestamp(end)

    data_slice = df.loc[slice_start:slice_end]
    diff_cols = find_diff_cols(data_slice)

    if params["use_differences"]:
        all_cols = diff_cols + params["mandatory_cols"]
    else:
        all_cols = [y for y in data_slice.columns if y not in diff_cols]

    data_array = []
    for col in all_cols:
        data_array.append(data_slice[col])

    if reshape:
        X = np.array(data_array).reshape(-1,len(all_cols))
    else:
        X = data_slice[all_cols]

    return X


def get_lagged_data_slice(df, start, end, params, reshape=True):
    '''
    Create a data slice for causal discovery algorithms
    from the gcastle package
    
    Args:
        df (DataFrame): Data
        start (str): Start date
        params (Namespace): Parameters

    Returns:
        X (np.array): Data slice
    '''

    data_slice = get_data_slice(df, start, end, params, reshape=False)

    # create lagged versions of all variables with max shift of days
    df_lagged = data_slice.copy()

    for i in range(1, params["lags"]+1):
        for col in data_slice.columns:
            df_lagged[f'{col}_lag{i}'] = data_slice[col].shift(i)

    # drop rows with NaNs
    df_lagged = df_lagged.dropna()

    if reshape:
        # create matrix with all columns
        X = np.array(df_lagged).reshape(-1,len(df_lagged.columns))

    else: # Return dataframe
        X = df_lagged
            

    return X


def load_dataset(filename):
    '''
    Load the dataset from a local file and set
    the date/time column as index

    Returns:
        df (DataFrame): Data
    '''

    # Load data
    df = pd.read_csv(filename, index_col=0, parse_dates=True)

    return df


def scale_dataframe(df, params):
    '''
    Scale the dataframe using the specified scaling type

    Args:
        df (DataFrame): Data
        params (Namespace): Parameters

    Returns:
        df_scaled (DataFrame): Scaled data
    '''

    scaler = sklearn.preprocessing.MinMaxScaler()

    if params["scaling_type"] == 'perc_range':
        scaler = sklearn.preprocessing.MinMaxScaler()

    elif params["scaling_type"]  == 'ngperc_range':
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-1,1))

    elif params["scaling_type"]  == 'full_range':
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(0,100))

    elif params["scaling_type"]  == 'ngfull_range':
        scaler = sklearn.preprocessing.MinMaxScaler(feature_range=(-100,100))
    else:
        print(f'\n[Warning] Unknown scaling type {params["scaling_type"]}. Using perc_range\n')

    # Apply scaler to data
    df_scaled = scaler.fit_transform(df)

    # Convert back to pandas dataframe
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)

    return df 


def mwcd(df, params, change_points, temporal, pbar):
    '''
    Run the MWCD algorithm using the gcastle package

    Args:
        df (DataFrame): Data
        params (Namespace): Parameters
        change_points (List): List of change points
        temporal (bool): Compute temporal causal matrix
        pbar (tqdm): Progress bar

    Returns:
        causal_matrices (List): List of causal matrices
    '''

    causal_matrices = []

    if torch.cuda.is_available():
        device_type = 'gpu'
    else:
        device_type = 'cpu'

    for i in range(0, len(change_points)):

        if i+1 < len(change_points):
            print(f"Computing Causal Matrix for Section: {change_points[i]} to {change_points[i+1]}")

            if params["causal_discovery_algo"] == 'DAG-GNN':

                if temporal:
                    X = get_lagged_data_slice(df, change_points[i], change_points[i+1], params)
                else:
                    X = get_data_slice(df, change_points[i], change_points[i+1], params)

                gnn = DAG_GNN(device_type=device_type)
                gnn.learn(X)

                # append causal matrix
                causal_matrices.append(gnn.causal_matrix)

            elif params["causal_discovery_algo"] == 'NoTearsNonLinear':

                if temporal:
                    X = get_lagged_data_slice(df, change_points[i], change_points[i+1], params)
                else:
                    X = get_data_slice(df, change_points[i], change_points[i+1], params)
                tqdm.write(f'[Info] Variables for Causal Discovery: {X.shape[1]}')
                notears = NotearsNonlinear(device_type=device_type)
                notears.learn(X)

                # append causal matrix
                causal_matrices.append(notears.causal_matrix)

            elif params["causal_discovery_algo"] == 'DirectLiNGAM':

                if temporal:
                    X = get_lagged_data_slice(df, change_points[i], change_points[i+1], params)
                else:
                    X = get_data_slice(df, change_points[i], change_points[i+1], params)
                tqdm.write(f'[Info] Variables for Causal Discovery: {X.shape[1]}')
                lingam = DirectLiNGAM()
                lingam.learn(X)

                # append causal matrix
                causal_matrices.append(lingam.causal_matrix)

            elif params["causal_discovery_algo"] == 'PCMCI':

                # get diff cols
                diff_cols = find_diff_cols(df)

                selected_cols = diff_cols + params["mandatory_cols"]
                tau_max = 10
                pc_alpha = 0.05

                if params["use_differences"]:
                    selected_cols = [y for y in df.columns if y not in diff_cols]

                df_slice = get_data_slice(df, change_points[i], change_points[i+1], params, reshape=False)
                
                X = df_slice[selected_cols].to_numpy()
                pcmci_df = pp.DataFrame(X, var_names=selected_cols)
                pcmci = PCMCI(dataframe=pcmci_df, cond_ind_test=ParCorr(significance='analytic'), verbosity=0)
                results = pcmci.run_pcmciplus(tau_min=0, tau_max=tau_max, pc_alpha=pc_alpha)

                # append causal matrix
                causal_matrices.append(results['val_matrix'])

            pbar.update(1)

    return causal_matrices


def find_diff_cols(df):
    '''
    Find columns with diff in the column name

    Args:
        df (DataFrame): Data

    Returns:
        diff_cols (List): List of columns with differences
    '''

    diff_cols = []

    for col in df.columns:
        if 'diff' in col:
            diff_cols.append(col)

    return diff_cols


def get_change_points(df, params):
    '''
    Compute change points in the data

    Args:
        df (DataFrame): Data
        params (Namespace): Parameters

    Returns:
        change_points (List): List of change points
    '''

    # prepare data
    input = df[params["target_variable"]].sort_index().values.reshape(-1,1)

    # compute change points
    algo = rpt.Pelt(model=params["model"], min_size=params["min-window"])
    algo.fit(input)
    result = algo.predict(pen=params["penalty"])

    split_dates = [df.index[0]]

    for bkp in result:
        if bkp < len(df.index):
            split_dates.append(df.index[bkp])
        elif bkp == len(df.index):
            split_dates.append(df.index[-1])

    change_points = pd.to_datetime(split_dates)

    if params["verbose"]:
        print(f'| {len(split_dates) -1} Change Point based windows:'.ljust(LINE_LENGTH, ' ') + '|')
        print(f'| Algorithm: Pruned Exact Linear Time (PELT)'.ljust(LINE_LENGTH, ' ') + '|') 
        print(f'| Model: {params["model"]}, Penalty: {params["penalty"]}, min_size: {params["min-window"]}'.ljust(LINE_LENGTH, ' ') + '|')
        print('+'.ljust(LINE_LENGTH, '-') + '+')

        for i in range(0, len(change_points)):
            if i+1 < len(change_points):
                print(f'| {change_points[i].strftime("%d.%m.%y")} to {change_points[i+1].strftime("%d.%m.%y")} - {(change_points[i+1] - change_points[i]).days} days'.ljust(LINE_LENGTH, " ") + '|')

        print('+'.ljust(LINE_LENGTH, '-') + '+')

    return change_points


def get_config():
    """
    Get configuration for the MWCD algorithm from the config.toml file
    """

    # Load config file
    with open("../config.toml", mode="rb") as fp:
        config = tomli.load(fp)

    # verfiy config
    if "data" not in config:
        raise ValueError("Missing data configuration in config.toml")
    else:
        if "target" not in config["data"]:
            raise ValueError("Missing target variable in data configuration in config.toml")
        if "path" not in config["data"]:
            raise ValueError("Missing path in data configuration in config.toml")
        if not os.path.exists(config["data"]["path"]):
                raise ValueError(f"File {config['data']['path']} not found. The base data folder is at ../data/. Please make sure the file exists.")
        if "scaling" not in config["data"]:
            raise ValueError("Missing scaling in data configuration in config.toml")
        if "use-differences" not in config["data"]:
            raise ValueError("Missing use-differences in data configuration in config.toml")
        if "always-include" not in config["data"]:
            raise ValueError("Missing always-include in data configuration in config.toml")

    if "causal-discovery" not in config:
        raise ValueError("Missing causal-discovery configuration in config.toml")
    else:
        if "lags" not in config["causal-discovery"]:
            raise ValueError("Missing lags in causal-discovery configuration in config.toml")
        if "algorithms" not in config["causal-discovery"]:
            raise ValueError("Missing algorithm in causal-discovery configuration in config.toml")
        
    if "change-point-detection" not in config:
        raise ValueError("Missing change-point-detection configuration in config.toml")
    else:
        if "window-sizes" not in config["change-point-detection"]:
            raise ValueError("Missing window-size in change-point-detection configuration in config.toml")
        if "model" not in config["change-point-detection"]:
            raise ValueError("Missing model in change-point-detection configuration in config.toml")
        if "penalty" not in config["change-point-detection"]:
            raise ValueError("Missing penalty in change-point-detection configuration in config.toml")
        
    if "misc" not in config:
        raise ValueError("Missing misc configuration in config.toml")
    else:
        if "verbose" not in config["misc"]:
            raise ValueError("Missing verbose in misc configuration in config.toml")
        if "label" not in config["misc"]:
            raise ValueError("Missing label in misc configuration in config.toml")

    return config



if __name__ == '__main__':

    # Load and verify config
    config = get_config()

    # create results folder if not existing
    if not os.path.exists('../results'):
        print(f'[INFO] Creating results folder...'.ljust(LINE_LENGTH, ' ') + '|')
        os.makedirs('../results')

    with all_logging_disabled():

        os.environ['CASTLE_BACKEND'] ='pytorch'
        print_start()

        df_raw = load_dataset(config["data"]["path"])

        all_scalings = ['perc_range']
        all_window_sizes = config["change-point-detection"]["window-sizes"]
        all_lags = config["causal-discovery"]["lags"]
        all_use_differences = config["data"]["use-differences"]
        all_algorithms = config["causal-discovery"]["algorithms"]
        verbose = config["misc"]["verbose"]
        target_variable = config["data"]["target"]
        pelt_model = config["change-point-detection"]["model"]
        penalty = config["change-point-detection"]["penalty"]
        scaling = config["data"]["scaling"]
        label = config["misc"]["label"]
        mandatory_cols = config["data"]["always-include"]

        for window_size in all_window_sizes:

            # Define parameters and print
            params = {
                "target_variable": target_variable,
                "model": pelt_model,
                "min-window": window_size,
                "verbose": verbose,
                "penalty": penalty,
                "mandatory_cols": mandatory_cols,
            }

            # Compute Change Points in unscaled data
            change_points = get_change_points(df_raw, params)
            
            for cd_algo in all_algorithms:
                for lags in all_lags:
                    for use_differences in all_use_differences:

                        # Define parameters and print
                        params["scaling_type"] = scaling
                        params["use_differences"] = use_differences
                        params["lags"] = lags
                        params["causal_discovery_algo"] = cd_algo

                        print_params(params)

                        with tqdm(total=(2*len(change_points)+2), desc='[INFO] Computing Causal Matrices') as pbar:

                            df_scaled = scale_dataframe(df_raw, params)

                            tqdm.write('[Info] Computing Causal Matrices for all sections...')
                            causal_matrices = mwcd(df_scaled, params, change_points, False, pbar)
                            np.save(f'../results/{label}-cm-sections-{cd_algo}.npy', causal_matrices)

                            # Compute temporal graph
                            tqdm.write('[Info] Computing Temporal Causal Matrices for all sections...')
                            temporal_matrices = mwcd(df_scaled, params, change_points, True, pbar)
                            np.save(f'../results/{label}-cm-temporal-sections-{cd_algo}.npy', temporal_matrices)

                            # Compute Causal Graph for all sections combined
                            tqdm.write('[Info] Computing Causal Matrix for complete time series...')
                            causal_matrix = mwcd(df_scaled, params, [change_points[0], change_points[-1]], False, pbar)
                            np.save(f'../results/{label}-cm-all-{cd_algo}.npy', causal_matrix)

                            # Compute Temporal Graph for all sections combined
                            tqdm.write('[Info] Computing Temporal Causal Matrix for complete time series...')
                            temporal_matrix = mwcd(df_scaled, params, [change_points[0], change_points[-1]], True, pbar)
                            np.save(f'../results/{label}-cm-temporal-all-{cd_algo}.npy', temporal_matrix)

    print('\n[INFO] Finished running MWCD.')
