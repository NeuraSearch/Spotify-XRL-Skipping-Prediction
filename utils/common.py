from lib import constants as consts

# a dictionary containing the state representation, its size, and whether it has been "corrected" (i.e., the temporal data leakage was removed)
def state_representation_information(no_temporal_leakage):
    if no_temporal_leakage:
        return {
            "state": consts.STATE,
            "state_size": consts.STATE_N_VARS,
            "corrected_state": True
        }
    else:
        return {
            "state": consts.TEMPORAL_LEAKAGE_STATE,
            "state_size": consts.TEMPORAL_LEAKAGE_STATE_N_VARS,
            "corrected_state": False
        }
