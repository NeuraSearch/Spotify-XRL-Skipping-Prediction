import os
import logging
import optuna
import numpy as np

from optuna.samplers import TPESampler

from lib.base_model import BaseModel
from lib.cli_parser import optimizer_parse_args

from utils import common

class ModelOptimizer(BaseModel):
    def __init__(self, num_episodes, model_name, dqn_type, network_type, no_temporal_leakage, test_set, n_trials, n_jobs):
        self.num_episodes = num_episodes
        self.model_name = model_name
        self.original_model_name = model_name
        self.dqn_type = dqn_type
        self.test_set = test_set
        self.network_type = network_type
        self.no_temporal_leakage = no_temporal_leakage

        self.n_trials = n_trials
        self.n_jobs = n_jobs

        # Random sampling is used instead of the TPE algorithm until the given
        # number of trials finish in the same study
        self.n_startup_trials = 4

        self.sampler_method = "TPE"
        self.sampler = TPESampler(n_startup_trials=self.n_startup_trials)

        # disable optuna logging if not in debug mode
        self.is_debug = (logging.getLogger().getEffectiveLevel() == logging.DEBUG) * 1
        if not self.is_debug:
            optuna.logging.set_verbosity(optuna.logging.WARNING)

    def run_optimization(self):
        logging.debug("Sampler: %s", self.sampler_method)

        # create an optuna study with specified sampler
        study = optuna.create_study(sampler=self.sampler)

        # run optimisation
        try:
            study.optimize(lambda trial: self.objective(trial),
                           n_trials=self.n_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            pass

        # get best trial and print its parameters
        trial = study.best_trial
        print("Number of finished trials: ", len(study.trials) - 1)
        print("Best trial has value:", -trial.value)
        print("Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # save study results to optimization log file
        self.__save_study_to_csv(study)

    def objective(self, trial):
        # fetch a set of parameters generated by trial
        model_params = self.__dqn_params(trial)
        # set the model's parameter keeping into consideration the trial number
        new_model_name = "{0}_Trial-{1}".format(self.original_model_name, trial.number + 1)

        # run the model 5 times
        _maas = []
        for x in range(5):
            super().__init__(self.num_episodes, new_model_name, x + 1, self.dqn_type, self.network_type, model_params["horizon"], self.no_temporal_leakage, self.test_set, is_optimisation=True)

            # train and then evaluate model
            try:
                self.train(**model_params)
            except AssertionError:
                raise optuna.exceptions.TrialPruned()

            # evaluate the mode. Don't save the results because it is not required when performing optimisation
            maa, first_pred_acc = self.evaluate(save_result=False)
            _maas.append(maa)

            # add custom information to the trial
            __info = common.state_representation_information(self.no_temporal_leakage)
            __info["model_path"] = self.agent_save_path
            __info["maa"] = maa
            __info["first_pred_acc"] = first_pred_acc
            for key, value in __info.items():
                trial.set_user_attr(key, value)

        print("Trial finished")

        # return negative average. Negative because that's how optuna's optimization works
        return -np.average(_maas)

    def __save_study_to_csv(self, study):
        # create dataframe of the trials
        dataframe = study.trials_dataframe()
        report_name = "opt_report-{}_sampler.csv".format(self.sampler_method)
        log_path = os.path.join(self.results_path, report_name)

        # drop "number" column because not meaningful
        dataframe.drop("number", axis=1, inplace=True)

        logging.debug("Writing report to %s", log_path)

        # convert dataframe to csv
        with open(log_path, "a+") as file:
            dataframe.to_csv(file, index=False, header=file.tell() == 0)

    # parameter selection
    def __dqn_params(self, trial):
        memory = trial.suggest_categorical("memory", [int(1e4), int(1e5)])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3)
        nstep = trial.suggest_int("horizon", 1, 4)
        discount = trial.suggest_categorical("discount", [0.9, 0.95, 0.98, 0.99])
        target_sync_frequency = trial.suggest_categorical("target_sync_frequency", [1, 500, 1000])
        
        # different types of network and their size
        network_size = trial.suggest_categorical("network_size", [32, 64, 128, 256])
        network_depth = trial.suggest_int("horizon", 1, 4)
        network_rnn_horizon = trial.suggest_int("network_rnn_horizon", 2, 6)
        if (self.network_type == "mlp") or (self.network_type == "mlp_obs_stacking"):
            network = { "type": "auto", "size": network_size, "depth": network_depth }
        elif self.network_type == "lstm":
            network = { "type": "auto", "size": network_size, "depth": network_depth, "rnn": network_rnn_horizon }
        elif self.network_type == "gru":
            network = []
            for i in range(network_depth):
                network += [dict(type="dense", size=network_size)]
            network += [dict(type="gru",  size=network_size, horizon=network_rnn_horizon)]

        return {
            "network": network,
            "memory": memory,
            "batch_size": batch_size,
            "update_frequency": batch_size, # update_frequency is the same as batch_size
            "start_updating": 1000,
            "learning_rate": learning_rate,
            "horizon": nstep,
            "discount": discount,
            "predict_terminal_values": False,
            "target_sync_frequency": target_sync_frequency,
            "config": {
                "buffer_observe": False
            }
        }

if __name__ == "__main__":
    args = optimizer_parse_args()

    ModelOptimizer(args.n_episodes, args.model_name, args.dqn_type, args.network, args.no_temporal_leakage, args.test_set, args.n_trials, args.n_jobs).run_optimization()