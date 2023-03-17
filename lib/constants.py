import enum
import re

TRAIN_DATA_PATH = "data/training_set/training_set.parquet"

_reason_end_list = {
    "hist_user_behavior_reason_end_appload" : "UB | RE | Appload",
    "hist_user_behavior_reason_end_backbtn" : "UB | RE | Backbtn",
    "hist_user_behavior_reason_end_clickrow" : "UB | RE | Clickrow",
    "hist_user_behavior_reason_end_clickside" : "UB | RE | Clickside",
    "hist_user_behavior_reason_end_endplay" : "UB | RE | Endplay",
    "hist_user_behavior_reason_end_fwdbtn" : "UB | RE | Fwdbtn",
    "hist_user_behavior_reason_end_logout" : "UB | RE | Logout",
    "hist_user_behavior_reason_end_popup" : "UB | RE | Popup",
    "hist_user_behavior_reason_end_remote" : "UB | RE | Remote",
    "hist_user_behavior_reason_end_trackdone" : "UB | RE | Trackdone",
    "hist_user_behavior_reason_end_uriopen" : "UB | RE | Uriopen"
}

STATE = [
    "session_position", "context_switch", "no_pause_before_play",
    "short_pause_before_play", "long_pause_before_play", "hist_user_behavior_n_seekfwd",
    "hist_user_behavior_n_seekback", "hist_user_behavior_is_shuffle", "hour_of_day",
    "premium", "duration", "release_year", "us_popularity_estimate", "acousticness",
    "beat_strength", "bounciness", "danceability", "dyn_range_mean", "energy", "flatness",
    "instrumentalness", "key", "liveness", "loudness", "mechanism", "organism",
    "speechiness", "tempo", "time_signature", "valence", "acoustic_vector_0",
    "acoustic_vector_1", "acoustic_vector_2", "acoustic_vector_3", "acoustic_vector_4",
    "acoustic_vector_5", "acoustic_vector_6", "acoustic_vector_7", "context_type_catalog",
    "context_type_charts", "context_type_editorial_playlist",
    "context_type_personalized_playlist", "context_type_radio", "context_type_user_collection",
    "hist_user_behavior_reason_start_appload", "hist_user_behavior_reason_start_backbtn",
    "hist_user_behavior_reason_start_clickrow", "hist_user_behavior_reason_start_clickside",
    "hist_user_behavior_reason_start_endplay", "hist_user_behavior_reason_start_fwdbtn",
    "hist_user_behavior_reason_start_playbtn", "hist_user_behavior_reason_start_popup",
    "hist_user_behavior_reason_start_remote", "hist_user_behavior_reason_start_trackdone",
    "hist_user_behavior_reason_start_trackerror", "hist_user_behavior_reason_start_uriopen",
    "mode_major", "mode_minor"
]

TEMPORAL_LEAKAGE_STATE = STATE.copy()
TEMPORAL_LEAKAGE_STATE.insert(1, "session_length")

_offset = 57
for i in _reason_end_list.keys():
    TEMPORAL_LEAKAGE_STATE.insert(_offset, i)
    _offset += 1

STATE_SHAP_ANALYSIS = [
    "CTX | Session Position", "UB | Playlist Switch", "UB | PA | No",
    "UB | PA | Short", "UB | PA | Long", "UB | SC | Num Seekfwd",
    "UB | SC | Num Seekback", "CTX | Shuffle", "CTX | Hour Of Day",
    "CTX | Premium", "CTN | TR | Duration", "CTN | TR | Release Year", "CTN | TR | Popularity", "CTN | TR | Acousticness",
    "CTN | TR | Beat Strength", "CTN | TR | Bounciness", "CTN | TR | Danceability", "CTN | TR | Dyn Range Mean", "CTN | TR | Energy", "CTN | TR | Flatness",
    "CTN | TR | Instrumentalness", "CTN | TR | Key", "CTN | TR | Liveness", "CTN | TR | Loudness", "CTN | TR | Mechanism", "CTN | TR | Organism",
    "CTN | TR | Speechiness", "CTN | TR | Tempo", "CTN | TR | Time Signature", "CTN | TR | Valence", "CTN | TR | Acoustic Vector 0",
    "CTN | TR | Acoustic Vector 1", "CTN | TR | Acoustic Vector 2", "CTN | TR | Acoustic Vector 3", "CTN | TR | Acoustic Vector 4",
    "CTN | TR | Acoustic Vector 5", "CTN | TR | Acoustic Vector 6", "CTN | TR | Acoustic Vector 7", "CTN | TR | Mode Major", "CTN | TR | Mode Minor", "CTX | PT | Catalog",
    "CTX | PT | Charts", "CTX | PT | Editorial Playlist",
    "CTX | PT | Personalized Playlist", "CTX | PT | Radio", "CTX | PT | User Collection",
    "UB | RS | Appload", "UB | RS | Backbtn",
    "UB | RS | Clickrow", "UB | RS | Clickside",
    "UB | RS | Endplay", "UB | RS | Fwdbtn",
    "UB | RS | Playbtn", "UB | RS | Popup",
    "UB | RS | Remote", "UB | RS | Trackdone",
    "UB | RS | Trackerror", "UB | RS | Uriopen"
]

TEMPORAL_LEAKAGE_STATE_SHAP_ANALYSIS = STATE_SHAP_ANALYSIS.copy()
TEMPORAL_LEAKAGE_STATE_SHAP_ANALYSIS.insert(1, "Session Length | CX | SL")

_offset = 57
for i in _reason_end_list.values():
    TEMPORAL_LEAKAGE_STATE_SHAP_ANALYSIS.insert(_offset, i)
    _offset += 1

STATE_N_VARS = len(STATE)
TEMPORAL_LEAKAGE_STATE_N_VARS = len(TEMPORAL_LEAKAGE_STATE)

def __create_dictionary(features):
    __dict = {}
    for i, val in enumerate(features):
        new_val = re.sub(r'(?:^|_)([a-z0-9])', lambda x: x.group(1).upper(), val)
        __dict[new_val] = i

    return __dict

STATE_VARIABLES = enum.Enum("STATE_VARIABLES", __create_dictionary(STATE))
TEMPORAL_LEAKAGE_STATE_VARIABLES = enum.Enum("TEMPORAL_LEAKAGE_STATE_VARIABLES", __create_dictionary(TEMPORAL_LEAKAGE_STATE))
