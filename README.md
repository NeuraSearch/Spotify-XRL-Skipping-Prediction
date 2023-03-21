# Spotify-XRL-Skipping-Prediction

An investigation on the utility of users’ historical data for the task of sequentially predicting users’ music skipping behaviour using Deep Reinforcement Learning. The analysis is performed on the [Spotify’s Music Streaming Sessions Dataset (MSSD)](https://www.aicrowd.com/challenges/spotify-sequential-skip-prediction-challenge).

This repository contains the source code for the approach outlined in the Full Paper [Why People Skip Music? On Predicting Music Skips using Deep Reinforcement Learning](https://dl.acm.org/doi/10.1145/3576840.3578312), accepted at the _2023 ACM SIGIR Conference on Human Information Interaction and Retrieval_ ([CHIIR2023](https://sigir.org/chiir2023)).

For the YouTube presentation of this paper, please click [here](https://www.youtube.com/watch?v=x8zJtZ4rLJc).

To know more about our research activities at NeuraSearch Laboratory, please follow us on Twitter ([@NeuraSearch](https://twitter.com/NeuraSearch)) and to get notified of future uploads please subscribe to our [YouTube channel](https://www.youtube.com/@neurasearch)! 


# Installation

The required Python packages can be found in `requirements.txt`. Using a package manager such as `pip`, they can be easily installed as follows:

```
conda create --name XRL-Skipping-Prediction python=3.9
conda activate XRL-Skipping-Prediction
pip3 install -r requirements.txt
```

# Data Preparation

**Part 1**. Upload the track features and the individual log files (from the MSSD) within the `data/` folder as follows:
- `data/track_features/` should contain:
    - `tf_000000000000.csv`
    - `tf_000000000001.csv`
- `data/training_set/` should contain:
    - `log_0_20180715_000000000000.csv`
    - `log_1_20180715_000000000000.csv`
    - `log_2_20180715_000000000000.csv`
    - `log_3_20180715_000000000000.csv`
- `data/test_set` should contain (e.g., to generate the **Test Set T1** in the accompanying paper):
    - `log_4_20180715_000000000000.csv`

**Part 2**:
```python
python3 data_preprocessing.py
```

The following files should have been created:
- `data/track_features/track_features_data.parquet`
- `data/training_set/training_set.parquet`
- `data/test_set/test_set.parquet`

# Deep Reinforcement Learning Model

## [Manual Execution](#manual-execution)

```
python3 model.py --model-name MyFirstDQN \
    --dqn-type dqn \
    --network mlp \
    --test-set data/test_set/test_set.parquet
```

The DQN is run five times on the given test set. The results and the models can be located at `/results` and `/results/models` respectively.

The `model.py` script will train and evaluate our proposed approach.

Below is the list of all command line parameters:
| Command | Description |
| --- | --- |
| `--model-name` | Name of the model (e.g., *MyFirstDQN*) |
| `-v` | [**Optional**]. Enable verbose printing |
| `-n` | [**Optional**]. Specify the number of episodes for training and evaluation. If not set, all episodes are considered |
| `--dqn-type` | [**Optional**, default: `dqn`]. Type of DQN: Vanilla DQN (`dqn`), Double DQN (`ddqn`), or Dueling DQN (`dueling_dqn`) |
| `--n-step` | [**Optional**, default: `1`]. n-step learning (>=1). |
| `--network` | [**Optional**, default: `mlp`]. Type of network: MLP (`mlp`), Observations Stacking (`mlp_obs_stacking`), LSTM (`lstm`), or GRU (`gru`) |
| `--no-temporal-leakage` | [**Optional**]. Do not remove the data leaking features. By default (i.e., no flag), such features are removed from the state representation |
| `--test-set` | The path to testing set (e.g., `data/testing_set/test_set.parquet`) |

### Post-Hoc (SHAP) analysis

By default, the post-hoc ([SHAP](https://github.com/slundberg/shap)) analysis is not performed given its extensive computational requirements. To perform a SHAP analysis, please make the following change in `model.py`:
```python
self.evaluate(shap_analysis=False)
```
:arrow_right:
```python
self.evaluate(shap_analysis=True)
```

Then, run `model.py` as outlined in Section [Manual Execution](#manual-execution). The results from the SHAP analysis can be found at `/results/shap`.

### Ablation Analysis of the State Representation

Details on how to perform the ablation analysis can be found within the `def __retrieve_df_and_n_episodes_from(self, data_path)` function in `lib/base_model.py`. This procedure requires some manual intervention.

By default (and with a "corrected" state representation), the following line is executed:
```python
df = np.delete(df, [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], 1)
```

These indexes refer to the **Reason End (RE)** User Behaviour (UB) features in the state representation, which is one of the two identified leaking features.

In order to remove further features, and hence perform the ablation analysis, the following two steps have to be performed:

**Step 1**. Locate the indexes of the features to be removed. They can be found in `lib/constants.py` and looking at the `STATE` variable. For example, if you wish to remove the pauses (i.e., `no_pause_before_play`, `short_pause_before_play`, and `long_pause_before_play`), the indexes are: 2, 3, 4.

**Step 2**. Change the previous block of code in `lib/base_model.py` as follows:
```python
df = np.delete(df, [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], 1)
```
:arrow_right:
```python
df = np.delete(df, [2, 3, 4, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], 1)
```

Please note that these indexes are for when interacting with the `STATE` variable, and therefore on a state without the temporal data leakage problem. The `TEMPORAL_LEAKAGE_STATE` refers to the original and not corrected state representation. Since it contains the `Session Length (SL)`, the indexes have to be adjusted accordingly.

## Hyperparameter Optimisation

```
python3 model_optimizer.py --model-name MyFirstOptDQN \
    --dqn-type dqn \
    --network mlp \
    --n-trials 20 \
    --n-jobs 2 \
    --test-set data/test_set/test_set.parquet
```

This script will perform hyperparameter optimisation on the given DQN type and network type. Given a set of parameters, the DQN is run five times on the given test set. This process, whereby a different set of parameters is selected, is repeated for `n-trials` times. The results and the models can be located at `/results` and `/results/opt` respectively.

Below is the list of all command line parameters:
| Command | Description |
| --- | --- |
| `--model-name` | Name of the model (e.g., *MyFirstDQN*) |
| `-v` | [**Optional**]. Enable verbose printing |
| `-n` | [**Optional**]. Specify the number of episodes for training and evaluation. If not set, all episodes are considered |
| `--dqn-type` | [**Optional**, default: `dqn`]. Type of DQN: Vanilla DQN (`dqn`), Double DQN (`ddqn`), or Dueling DQN (`dueling_dqn`) |
| `--n-step` | [**Optional**, default: `1`]. n-step learning (>=1). |
| `--network` | [**Optional**, default: `mlp`]. Type of network: MLP (`mlp`), Observations Stacking (`mlp_obs_stacking`), LSTM (`lstm`), or GRU (`gru`) |
| `--no-temporal-leakage` | [**Optional**]. Do not remove the data leaking features. By default (i.e., no flag), such features are removed from the state representation |
| `--n-trials` | [**Optional**, default: `10`]. Number of trials |
| `--n-jobs` | [**Optional**, default: `1`] Number of jobs for parallel execution |
| `--test-set` | The path to testing set (e.g., `data/testing_set/test_set.parquet`) |

# Cite
Please, cite this work as follows:

```
@inproceedings{10.1145/3576840.3578312,
  author = {Meggetto, Francesco and Revie, Crawford and Levine, John and Moshfeghi, Yashar},
  title = {Why People Skip Music? On Predicting Music Skips Using Deep Reinforcement Learning},
  year = {2023},
  isbn = {9798400700354},
  publisher = {Association for Computing Machinery},
  address = {New York, NY, USA},
  url = {https://doi.org/10.1145/3576840.3578312},
  doi = {10.1145/3576840.3578312},
  booktitle = {Proceedings of the 2023 Conference on Human Information Interaction and Retrieval},
  pages = {95–106},
  numpages = {12},
  keywords = {Spotify, Deep Reinforcement Learning, Music, Prediction, User Behaviour, Skipping},
  location = {Austin, TX, USA},
  series = {CHIIR '23}
}
```

```
Francesco Meggetto, Crawford Revie, John Levine, and Yashar Moshfeghi. 2023. Why People Skip Music? On Predicting Music Skips using Deep Reinforcement Learning. In Proceedings of the 2023 Conference on Human Information Interaction and Retrieval (CHIIR '23). Association for Computing Machinery, New York, NY, USA, 95–106. https://doi.org/10.1145/3576840.3578312
```