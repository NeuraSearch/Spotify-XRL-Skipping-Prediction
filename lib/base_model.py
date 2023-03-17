import datetime
import time
import os
import logging
import json
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner

from lib.custom_evaluation import CustomEvaluation
from lib import constants as consts
from utils import common

import DRL

# no need for GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class BaseModel:
    def __init__(self, num_episodes, model_name, run_number, dqn_type, network_type, n_step, no_temporal_leakage, test_set, is_optimisation=False):
        self.num_episodes = num_episodes
        self.model_name = model_name
        self.run_number = run_number
        self.dqn_type = dqn_type
        self.network_type = network_type
        self.n_step = n_step
        self.no_temporal_leakage = no_temporal_leakage

        # create folder for the model and paths
        self.save_name = "{0}_{1}_{2}".format(self.model_name, self.run_number, time.strftime("%Y%m%d-%H%M%S"))
        self.results_path = "results/"
        if is_optimisation:
            model_folder = "opt"
        else:
            model_folder = "models"
        self.base_save_path = os.path.join(self.results_path, model_folder, self.save_name)

        self.agent_info_path = os.path.join(self.base_save_path, "information.json")
        self.agent_save_path = os.path.join(self.base_save_path, "model")
        os.makedirs(os.path.dirname(self.agent_save_path), exist_ok=True)

        # training and testing dataset paths
        self.train_data_path = consts.TRAIN_DATA_PATH
        self.test_data_path = test_set

        # flag if verbose or not
        self.is_debug = (logging.getLogger().getEffectiveLevel() == logging.DEBUG) * 1

    def train(self, **model_params):
        logging.debug("\Training Model")
        df, df_Y, num_episodes = self.__retrieve_df_and_n_episodes_from(self.train_data_path)

        print("{0}\t\tStarting learning".format(datetime.datetime.now()))

        # create environment and agent with model parameters
        env = self.__create_environment(df, df_Y, is_train=True)
        agent = Agent.create(agent=self.dqn_type, environment=env, **model_params)

        # save model information to `information.json`
        self.__save_model_information(**model_params)

        start_time = time.time()

        # start learning via Runner interface
        runner = Runner(agent=agent, environment=env)
        runner.run(num_episodes=num_episodes, use_tqdm=self.is_debug)
        runner.close()

        # save agent and then close instances
        agent.save(directory=self.agent_save_path)
        agent.close()
        env.close()

        # log stats on output
        total_time_multi = time.time() - start_time
        hours, rem = divmod(total_time_multi, 3600)
        minutes, seconds = divmod(rem, 60)
        logging.debug("Took {:0>2}:{:0>2}:{:05.2f} - {:.2f} FPS".format(
            int(hours), int(minutes), seconds, num_episodes / total_time_multi))

        print("{0}\t\tFinished learning".format(datetime.datetime.now()))

    def evaluate(self, shap_analysis=False, save_result=True):
        logging.debug("\nEvaluating Model")
        df, df_Y, num_episodes = self.__retrieve_df_and_n_episodes_from(self.test_data_path)

        print("{0}\t\tStarting evaluation".format(datetime.datetime.now()))

        # create environment and load agent from `self.agent_save_path`
        env = self.__create_environment(df, df_Y, is_train=False)
        agent = Agent.load(directory=self.agent_save_path)

        # perform evaluation
        custom_eval = CustomEvaluation(agent, env, df_Y, num_episodes, self.no_temporal_leakage, shap_analysis=False)
        maa, first_pred_acc, _ = custom_eval.run(self.base_save_path)

        # close evaluation environment
        env.close()

        # perform shap analysis if set
        if shap_analysis:
            # create environment
            env = self.__create_environment(df, df_Y, is_train=False)
            # perform shap evaluation
            custom_eval = CustomEvaluation(agent, env, df_Y, 50, self.no_temporal_leakage, shap_analysis=True)
            _, _, states = custom_eval.run(self.base_save_path)
            # perform shap analysis
            self.__perform_shap_analysis(df, df_Y, agent, env, states)
            # close shap environment
            env.close()

        # close agent
        agent.close()

        print("{0}\t\tFinished evaluation".format(datetime.datetime.now()))

        # if results want to be saved, save the model's performance in `results.csv`
        if save_result and (not shap_analysis):
            results_path = os.path.join(self.results_path, "results.csv")
            vars_dict = {
                "model": self.model_name, "run": self.run_number,
                "maa": maa, "first_pred_acc": first_pred_acc,
                "model_path": self.agent_save_path, "testing_set": self.test_data_path,
                "timestamp": datetime.datetime.now()
            }
            custom_eval.save_evaluation_to_csv(results_path, **vars_dict)
            print("Mean Average Accuracy: {}".format(maa))
            print("First Prediction Accuracy: {}".format(first_pred_acc))

        return maa, first_pred_acc

    def __perform_shap_analysis(self, df, df_Y, agent, env, states):
        # use Kernel Explainer and generate shap values
        def f(X):
            actions_arr = []
            states = env.reset()
            internals = agent.initial_internals()
            for i in range(len(X)):
                states = X[i]
                _actions, internals = agent.act(
                    states=states, internals=internals,
                    independent=True, deterministic=True
                )
                actions_arr.append(_actions)
            return np.array(actions_arr)

        explainer = shap.KernelExplainer(f, states)
        shap_values = explainer.shap_values(states, nsamples=200)

        # shap values can be loaded with `np.load('values.npy')`
        # save the computed shap values
        np.save(os.path.join(self.results_path, "shap", "SHAP-values-{0}.npy".format(self.save_name)), shap_values)

        # create the two types of summary plots and save as pdf and png
        if self.no_temporal_leakage:
            feature_names = consts.STATE_SHAP_ANALYSIS
        else:
            feature_names = consts.TEMPORAL_LEAKAGE_STATE_SHAP_ANALYSIS

        plt.figure()
        shap.summary_plot(shap_values, states, feature_names=feature_names, plot_type='bar')
        plt.gcf().set_size_inches(10, 6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "SHAP-bar-{0}.pdf".format(self.save_name)))
        plt.savefig(os.path.join(self.results_path, "SHAP-bar-{0}.png".format(self.save_name)))

        plt.figure()
        shap.summary_plot(shap_values, states, feature_names=feature_names)
        plt.gcf().set_size_inches(14, 6)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_path, "SHAP-summary-{0}.pdf".format(self.save_name)))
        plt.savefig(os.path.join(self.results_path, "SHAP-summary-{0}.png".format(self.save_name)))

    def __retrieve_df_and_n_episodes_from(self, data_path):
        # get dataframe from parquet file, calculate number of episodes (override or total) and
        # reduce dataframe by removing unnecessary columns
        df = pd.read_parquet(data_path).to_numpy()

        # take into consideration if a parameter from command line was the number of episodes. If no parameter was passed, then all episodes are used
        num_episodes = self.__adjust_num_episodes(self.num_episodes, len(np.unique(df[:, 0])))

        # delete first column --> session_id. Not useful
        df = np.delete(df, 0, 1)

        # all skip variables are the target Y (plus session_length which is used in environment to perform correct evaluation)
        # all the rest is X. Indexes:
        ## 0: session_length
        ## 1: skip_1
        ## 2: skip_2
        ## 3: skip_3
        ## 4: not_skipped
        if self.no_temporal_leakage:
            df_Y = df[:, 1:6]
            df = np.delete(df, [1, 2, 3, 4, 5], 1)

            # remove hist_user_behavior_reason_end_*
            df = np.delete(df, [58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], 1)
            ### if instead, you wish to perform the ablation analysis and remove, let's say, the pauses, then the previous line becomes:
            ### (to check the index, please refer to `lib/constants.py` and in particular the `STATE` variables. The pauses have index 2,3,4)
            # df = np.delete(df, [2, 3, 4, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68], 1)
        else:
            df_Y = df[:, 3:4]
            df = np.delete(df, [2, 3, 4, 5], 1)

        return df, df_Y, num_episodes

    # helper function to create the environment
    def __create_environment(self, df, df_Y, is_train=True):
        obs_stacking = self.network_type == "mlp_obs_stacking"
        return Environment.create(environment="gym", level="skip_prediction-v0", df=df, df_Y=df_Y, is_train=is_train, obs_stacking=obs_stacking, no_temporal_leakage=self.no_temporal_leakage)

    # function to create the `information.json` file which contains information about the state representation
    # and the model used
    def __save_model_information(self, **model_params):
        logging.debug("\nSaving Model Information to %s", self.agent_info_path)
        __info = common.state_representation_information(self.no_temporal_leakage)
        __info = {**__info, **model_params}
        __info["dqn_type"] = self.dqn_type
        __info["network_type"] = self.network_type
        __info["n_step"] = self.n_step

        with open(self.agent_info_path, "w") as json_file:
            json.dump(__info, json_file)

    # helper function that returns the number of episodes to consider
    def __adjust_num_episodes(self, n_episodes, total_episodes):
        if n_episodes == -1:
            num_episodes = total_episodes
        else:
            num_episodes = n_episodes

        return num_episodes
