# MWCD
MWCD: Multi-Window Causal Discovery Framework for Rock Slope Stability Analysis.

![MWCD Overview](figures/MWCD_overview.png)

## Get Started

In order to run the example, first install the [required packages](#required-packages), then run the [Example](#example) using the provided example dataset.

To customise the MWCD, use any valid combination of command line [options](#options).

If you use your own dataset, the following requirements have to be met:
- data has to be provided in a CSV file with the first row containing the variable names. See the example dataset for reference.
- (in this version) augmentation has to be done before running the MWCD algorithm, i.e. there is no automated computation of Freeze-Thaw-Cycles or negative-degree days currently implemented. All variables must be supplied as columns in the csv file.
- no pre-scaling required, the MWCD normalises all inputs

### Required Packages

To install the required Python packages, run `pip install -r requirements.txt` to install the packages specified in the `requirements.txt` file. Furthermore, the `gcastle` package is set to use PyTorch as backend. Install PyTorch [using the PyTorch Get Started instructions](https://pytorch.org/get-started/locally/).

### Example

To run the example, execute `python main.py`. The main script will load the example dataset from `/data` and use DAG-GNN for Causal Discovery. If you would like to tweak the algorithm or change the Causal Discovery algorithm, use any of the options specified under [Options](#options).


## Options

The MWCD offers a variety of options to adapt it to your dataset. Additionally, it is possible to pass a list of options to test with all possible combinations. The options are read from the `config.toml` file, which is a file in [TOML format](https://github.com/toml-lang/toml) describing the configuration used to run the MWCD.

All available options are listed below.

### Causal Discovery

Options to configure the causal discovery. Options are listed under [causal-discovery]

#### Algorithm

There are three different Causal Discovery algorithms available to run MWCD that can be selected by adding them as strings to the `algorithms` list. Available algorithms:
- DAG-GNN
- NoTearsNonLinear
- DirectLiNGAM
- PCMCI
- LPCMCI

Example to run the MWCD with DAG-GNN and again with NoTearsNonLinear: 
```toml
[causal-discovery]
algorithms = ["DAG-GNN", "NoTearsNonLinear"]
```

#### Lags

The number of lags for the shifting before temporal causal discovery can be set using the option for `lags`. 

Example to run the MWCD with 7 sample shifts and 14 sample shifts:
```toml
[causal-discovery]
lags = [7, 14]
```

### Change Point Detection

The MWCD uses Linearly penalized segmentation for Change Point Detection [Ruptures Documentation](https://centre-borelli.github.io/ruptures-docs/user-guide/detection/pelt/). Change Point Detection is listed under [change-point-detection]


#### Min Window Size

A minimum window size can be specified for the change point detection. The default value is 30 points. This should be adapted to the resolution of the dataset as it only specifies the minimum distance between change points in number of samples. Multiple window-sizes can be added to run in one execution to the `window-sizes` option.

Example to run the MWCD with two different window sizes, one of 14 and of of 30 days:
```toml
[change-point-detection]
window-sizes = [14, 30]
```

#### Penalty

To return the optimal breakpoints the signal is fit, where the fitting requires a penalty. The default penalty used in the MWCD is 2. To specify a different penalty, use the option `penalty`.

Example to run the MWCD Change Point Detection with a penalty of 5:
```toml
[change-point-detection]
penalty = 5
```

#### Model

PELT can be used with different segment models. Available models are "l1", "l2" and "rbf". The default model is "rbf". To use a different model, use the option `model` with the model name.

Example to run the MWCD Change Point Detection with the l2 model:
```toml
[change-point-detection]
model = "l2"
```

### Data

Options related to the dataset are listed under [data]

#### Path

To use a different dataset, provide the path under the option `path`. The path should be relative to the main script, i.e. if placed in the `data` folder: `../data/mydata.csv`.

Example to run the MWCD with data from the file `my_data.csv` in the data folder:
```toml
[data]
path = "../data/my_data.csv"
```

#### Target Variable

To specify a target variable for the change point detection, use the option `target`. The target name must correspond to one of the column names in the dataset file. 

For example, to use _block4a_ with the example dataset:
```toml
[data]
path = "../data/stampa.csv"
target = "block4a"
```

#### Scaling

To specify how to scale the data before passing it to the causal discovery algorithm, you may modify the option `scaling`. Available scalings are: 
- "perc_range":  scales all variables to a range of [0,1]
- "ngperc_range": scales all variables to a range of [-1,1]
- "full_range": scales all variables to a range of [0,100]
- "ngfull_range": scales all variables to a range of [-100,100]
We recommend the usage of the standard "perc_range" scaling, which is the default scaling.

For example, to use "full_range" scaling with the example dataset:
```toml
[data]
scaling = "full_range"
```

#### use-differences

We recommend to use displacement, soil moisture and infiltration differences instead of the original time series. If included in the dataset, they should labeled accordingly. If `use-differences` is set to true, the MWCD will automatically look for columns that contain the string "diff" and perform the causal discovery on these columns accordingly. To add columns to the causal discovery regardless of their naming, they have to be specified under [`always-include`](#always-include).
If `use-differences` is set to false, all columns that do not contain the keyword "diff" will be used for the causal discovery.

For example, to set `use-differences` to false:
```toml
[data]
use-differences = [false]
```

Since this option is also provided as a list, it is possible to add both options, true and false, to the list and run the MWCD once with differences and once without.

#### always-include

The list specified under `always-include` must contain the column names from the dataset that should be considered in the causal discovery regardless of the value in `use-differences`. 

For example, to always include the columns named `precipitation`, `temperature` and `frost`:
```toml
[data]
always-include = [
    "precipitation",
    "temperature",
    "frost"
    ]
```

### Misc

Under [misc], a collection of useful options is provided. 

#### verbose

The option `verbose` can be set to true to provide output on the command line. Alternatively, the output can be suppressed setting it to false.

For example, to set `verbose` to false:
```toml
[misc]
verbose = false
```

#### label

The option `label` enables you to change the naming scheme for the result files. These files are stored as `.npy` files with the following naming scheme:
```
<label>-cm-<MWCD part>-<causal discovery algorithm>.npy
```
where cm stands for Causal Matrix.

For example to set the label to `my_data_test`:
For example, to set `verbose` to false:
```toml
[misc]
label = "my_data_test"
```

This example would generate 4 files in the results folder when used with only DAG-GNN:
- `my_data_test-cm-all-DAG-GNN.npy`
- `my_data_test-cm-sections-DAG-GNN.npy`
- `my_data_test-cm-temporal-all-DAG-GNN.npy`
- `my_data_test-cm-temporal-sections-DAG-GNN.npy`



## Utilities

The output of the MWCD is per default a collection of adjacency matrices. These matrices are in a folder named `results` which is automatically created if it does not already exist. We provide utility functions to plot causal graphs from the Causal Discovery matrix output in the file `src/utils.py`.