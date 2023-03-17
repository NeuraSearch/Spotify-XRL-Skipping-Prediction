import os
import glob
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# list of Track (TR) features
TRACK_FEATURES_LIST = [
    "release_year", "duration", "us_popularity_estimate", "acousticness", "beat_strength",
    "bounciness", "danceability", "dyn_range_mean", "energy", "flatness", "instrumentalness", "key",
    "liveness", "loudness", "mechanism", "organism", "speechiness", "tempo",
    "valence", "acoustic_vector_0", "acoustic_vector_1", "acoustic_vector_2", "acoustic_vector_3",
    "acoustic_vector_4", "acoustic_vector_5", "acoustic_vector_6", "acoustic_vector_7"
]

# list of features to one-hot encode
FEATURES_TO_ENCODE = [
    "context_type", "hist_user_behavior_reason_start", "hist_user_behavior_reason_end"
]

# paths for location of files
TRACK_FEATURES_PATH = "data/track_features/track_features_data.parquet"

TRAINING_PATH = "data/training_set/"
TRAINING_SET_PATH = TRAINING_PATH + "training_set.parquet"

TEST_PATH = "data/test_set/"
TEST_SET_PATH = TEST_PATH + "test_set.parquet"

# merge the two track features files (tf_000000000000 and tf_000000000001) into one single dataframe
def merge_track_features():
    track_features_data_0 = pd.read_csv("data/track_features/tf_000000000000.csv")
    track_features_data_1 = pd.read_csv("data/track_features/tf_000000000001.csv")
    df = pd.concat([track_features_data_0, track_features_data_1])

    # scale all `TRACK_FEATURES_LIST` features to 0 mean and 1 standard deviation
    df[TRACK_FEATURES_LIST] = StandardScaler().fit_transform(df[TRACK_FEATURES_LIST])

    # one-hot encoding for mode (major and minor)
    df = pd.get_dummies(df, columns = ['mode'], prefix = ['mode'])

    # save track_features file as parquet
    __save_to_parquet(df, TRACK_FEATURES_PATH)

    del df

def __pre_process_files(files_path):
    # read in the track features dataframe
    track_features_df = pd.read_parquet(TRACK_FEATURES_PATH)

    # for every log file (in csv format) in folder `files_path`, perform processing
    # processing:
    # 1. read log file
    # 2. rename `track_id_clean` to `track_id`
    # 3. merge log file with the track features dataframe (`track_features_df`)
    # 4. drop `track_id` since it is of no use anymore
    # 5. perform pre_processing (--> __pre_process_data(...))
    # 6. save processed log file
    log_files = glob.glob(files_path + "*.csv")
    for idx, log_file in enumerate(log_files):
        print("Processing file number: {}".format(idx))
        df = pd.read_csv(log_file)
        # merge log_file and track metadata based on `track_id`
        df = df.rename(columns={"track_id_clean": "track_id"})
        df = pd.merge(df, track_features_df, how="left")
        df.drop("track_id", axis=1, inplace=True)

        # data pre-processing of features
        df = __pre_process_data(df)

        # save processed log
        print("Saving file number: {}".format(idx))
        __save_to_parquet(df, files_path + "processed_" + str(idx) + ".parquet")

def __merge_processed_logs(files_path, set_file_path):
    # get all processed log files from `files_path` (from the previous step --> __pre_process_files(...))
    log_files = sorted(glob.glob(files_path + "processed_*.parquet"), key=os.path.getmtime)
    # merge all the logs together into a single dataframe. We construct a single dataframe (e.g., training or test set)
    df = pd.concat([pd.read_parquet(fp) for fp in log_files], ignore_index=True)

    # save as parquet file
    __save_to_parquet(df, set_file_path)

    # delete the intermediary processing files (the ones generated in __pre_process_files(...))
    for log_file in log_files:
        os.remove(log_file)

def __process_session_id(file_path):
    # read in the parquet dataframe
    df = pd.read_parquet(file_path)

    # create mapping of session_ids, starting from 0. This is to reduce memory usage since originally the session_id is a long hash
    _dict = {k: v for v, k in enumerate(df["session_id"].unique())}
    df["session_id"] = np.vectorize(_dict.get)(df.session_id)

    # save dataframe with updated session_id field
    __save_to_parquet(df, file_path)

    return df

# save pickle to reduce space
def __save_to_parquet(df, file_path):
    df.to_parquet(file_path)

# this is the main function that performs the preprocessing of the data (UB and CX categories.
# The CN (TR) category has already been processed by merge_track_features()
def __pre_process_data(df):
    # drop `date` because not useful
    df.drop("date", axis=1, inplace=True)

    # convert all the boolean features to integer representation
    df["skip_1"] = df["skip_1"].astype(int)
    df["skip_2"] = df["skip_2"].astype(int)
    df["skip_3"] = df["skip_3"].astype(int)
    df["not_skipped"] = df["not_skipped"].astype(int)
    df["hist_user_behavior_is_shuffle"] = df["hist_user_behavior_is_shuffle"].astype(int)
    df["premium"] = df["premium"].astype(int)

    # one-hot encoding of the `FEATURES_TO_ENCODE` features
    for feature in FEATURES_TO_ENCODE:
        df = __encode_and_bind(df, feature)

    return df

# this helper method performs the one-hot encoding of the categorical features
def __encode_and_bind(original_dataframe, feature_to_encode):
    # manually input categories for encoding.
    # Some logs may not have all values, and this may cause inconsistencies
    # So we provide the list of all possible values and fill the missing ones with 0s
    dummies = pd.get_dummies(original_dataframe[[feature_to_encode]])
    if feature_to_encode == "context_type":
        all_values = ["context_type_catalog", "context_type_charts", "context_type_editorial_playlist", "context_type_personalized_playlist", "context_type_radio", "context_type_user_collection"]
    elif feature_to_encode == "hist_user_behavior_reason_start":
        all_values = ["hist_user_behavior_reason_start_appload", "hist_user_behavior_reason_start_backbtn", "hist_user_behavior_reason_start_clickrow", "hist_user_behavior_reason_start_clickside", "hist_user_behavior_reason_start_endplay", "hist_user_behavior_reason_start_fwdbtn", "hist_user_behavior_reason_start_playbtn", "hist_user_behavior_reason_start_popup", "hist_user_behavior_reason_start_remote", "hist_user_behavior_reason_start_trackdone", "hist_user_behavior_reason_start_trackerror", "hist_user_behavior_reason_start_uriopen"]
    elif feature_to_encode == "hist_user_behavior_reason_end":
        all_values = ["hist_user_behavior_reason_end_appload", "hist_user_behavior_reason_end_backbtn", "hist_user_behavior_reason_end_clickrow", "hist_user_behavior_reason_end_clickside", "hist_user_behavior_reason_end_endplay", "hist_user_behavior_reason_end_fwdbtn", "hist_user_behavior_reason_end_logout", "hist_user_behavior_reason_end_popup", "hist_user_behavior_reason_end_remote", "hist_user_behavior_reason_end_trackdone", "hist_user_behavior_reason_end_uriopen"]

    dummies = dummies.reindex(columns=all_values, fill_value=0)

    # add the encoded features and drop the original features
    res = pd.concat([original_dataframe, dummies], axis=1)
    res = res.drop([feature_to_encode], axis=1)

    return res

def process_files(general_path, dataframe_path):
    # process individual logs in `general_path`
    __pre_process_files(general_path)
    # merge processed logs from previous steps together, to form a single dataset
    __merge_processed_logs(general_path, dataframe_path)
    # vectorize `session_id` to lower values to reduce space
    __process_session_id(dataframe_path)

def main():
    # `merge_track_features()` to be performed only once, no need for multiple use
    print("Merge track features")
    merge_track_features()

    # process the training files to create a single training_set
    print("\nProcess training files")
    process_files(TRAINING_PATH, TRAINING_SET_PATH)

    # process the test files to create a single test_set
    # In the accompanying paper, we mention 5 different test sets. This can be achieved by running the data_processing.py file 5 times,
    # where each time the corresponding log files are provided in TEST_PATH. After each run, the resulting `test_set.parquet` can be renamed
    # meaningfully. For the next run, repeat the same procedure (i.e., put the original csv log files in the test folder, and run this script)
    # After running this script 5 times, you will have the 5 test sets.
    # Please note that by using this approach, it is also possible to have test sets that compromise of multiple log giles (as, for example, done for the training set)
    print("\nProcess test files")
    process_files(TEST_PATH, TEST_SET_PATH)

if __name__ == "__main__":
    main()
