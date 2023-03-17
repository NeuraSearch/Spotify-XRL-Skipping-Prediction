import argparse
import logging

def optimizer_parse_args():
    parser = __create_common_parser()

    parser.add_argument("--n-trials", help="Number of trials for optimizing hyperparameters", type=int, default=10)
    parser.add_argument("--n-jobs", help="Number of jobs", type=int, default=1)

    return __parse_and_set_logging_from_verbose(parser)

def parse_args():
    parser = __create_common_parser()

    return __parse_and_set_logging_from_verbose(parser)

def __create_common_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-v", "--verbose", help="Increase output verbosity", action="store_true")
    parser.add_argument("-n", "--n-episodes", help="Overwrite the number of episodes", default=-1, type=int)
    parser.add_argument("--model-name", help="Model name (e.g., 'MyFirstDQN')", required=True)

    # DQN specifics: dqn model type, n-step learning, and network type
    parser.add_argument("--dqn-type", choices=["dqn", "ddqn", "dueling_dqn"], default="dqn", help="Type of DQN model: Vanilla DQN (dqn), Double DQN (ddqn), or Dueling DQN (dueling_dqn)")
    parser.add_argument("--n-step", help="n-step learning (>=1). Default is 1", default=1, type=int)
    parser.add_argument("--network", choices=["mlp", "mlp_obs_stacking", "lstm", "gru"], default="mlp", help="Model architecture: MLP (mlp), Observations Stacking (mlp_obs_stacking), LSTM (lstm), or GRU (gru)")
    parser.add_argument("--no-temporal-leakage", help="Remove the temporal data leakage problem", action="store_true")

    parser.add_argument("--test-set", help="Location to testing dataset", required=True)

    return parser

def __parse_and_set_logging_from_verbose(parser):
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    return args
