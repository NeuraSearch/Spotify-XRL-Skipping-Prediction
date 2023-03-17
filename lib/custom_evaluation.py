import os
import logging
import json
import pandas as pd
import numpy as np

from lib import constants as consts

class CustomEvaluation:
    def __init__(self, agent, env, df_Y, num_episodes, no_temporal_leakage, shap_analysis=False):
        self.agent = agent
        self.env = env
        # df_Y is the target --> the ground truth --> the skipping activity as reported in the dataset
        self.df_Y = df_Y
        self.num_episodes = num_episodes
        self.no_temporal_leakage = no_temporal_leakage
        self.shap_analysis = shap_analysis

    def run(self, base_save_path):
        # use agent to generate predictions and also create ground_truth for
        # 2nd half of every session
        submission, ground_truth, states = self.__generate_model_submission()

        # if we are performing a SHAP analysis, we only need the states and not the metrics
        if self.shap_analysis:
            return 0, 0, np.array(states)

        # save submission and ground_truth in txt files
        self.__save_to_txt_file(submission, os.path.join(base_save_path, "submission.txt"))
        self.__save_to_txt_file(ground_truth, os.path.join(base_save_path, "ground_truth.txt"))

        # generate resulting metrics
        maa, first_pred_acc = self.__generate_metrics(submission, ground_truth, base_save_path)

        # return performance measures and the array of states (used by SHAP)
        return maa, first_pred_acc, states

    # function to save the current evalution to the `results.csv` file which contains all the evaluations
    def save_evaluation_to_csv(self, results_path, **vars_dict):
        # extract data from `agent.json`
        file = open(os.path.join(vars_dict["model_path"], "agent.json"))
        data = json.load(file)
        file.close()

        # convert json object to Pandas dataframe, add extra information.
        df = pd.json_normalize(data)

        # reorganise columns for consistent formatting
        offset = 29
        col_name = "config.buffer_observe"
        col_data = df.pop(col_name)
        df.insert(offset, col_name, col_data)

        i = 0
        for key, value in vars_dict.items():
            df.insert(loc=i, column=key, value=value)
            i += 1

        logging.debug("Writing evaluation results to %s", results_path)

        # append to `results_path` --> the results.csv file
        with open(results_path, "a+") as file:
            df.to_csv(file, index=False, header=file.tell() == 0)

    # https://github.com/crowdAI/skip-prediction-challenge-starter-kit/blob/master/local_evaluation.ipynb
    def __ave_acc(self, submission, ground_truth, counter):
        s = 0.0
        t = 0.0
        c = 1.0
        for x, y in zip(submission, ground_truth):
            if x != 0 and x != 1:
                raise Exception("Invalid prediction in line {}, should be 0 or 1".format(counter))
            if x == y:
                s += 1.0
                t += s / c
            c += 1
        return t / len(ground_truth)

    # https://github.com/crowdAI/skip-prediction-challenge-starter-kit/blob/master/local_evaluation.ipynb
    def __generate_metrics(self, submission, ground_truth, base_save_path):
        accuracies_file = open(os.path.join(base_save_path, "accuracies.txt"), "w")

        aa_sum = 0.0
        first_pred_acc_sum = 0.0
        counter = 0
        for sub, tru in zip(submission, ground_truth):
            if len(sub) != len(tru):
                raise Exception("Line {} should contain {} predictions, but instead contains "
                                "{}".format(counter + 1, len(tru), len(sub)))
            av_acc = self.__ave_acc(sub, tru, counter)

            # save av_acc in file to keep track of accuracies of each episode
            accuracies_file.write(str(av_acc) + "\n")

            aa_sum += av_acc
            first_pred_acc_sum += sub[0] == tru[0]
            counter += 1

        maa = aa_sum / counter
        first_pred_acc = first_pred_acc_sum / counter

        accuracies_file.close()

        return maa, first_pred_acc

    def __generate_model_submission(self):
        # prediction values/predictions and ground_truth values/predictions. Build them together
        # to speed up process since the dataframe is being iterated by the agent neverthless
        pr_vals = []
        pr_preds = []
        gt_vals = []
        gt_preds = []

        # list of the observed states
        pr_states = []

        i = 0
        # iterate through all sessions. Let agent make best possible decision, and store in
        # output the predicted actions (i.e., only of the second half of each session)
        for _ in range(self.num_episodes):
            states = self.env.reset()
            internals = self.agent.initial_internals()
            terminal = False
            while not terminal:
                # get curr_session_position and curr_session_length. Depending on whether we have temporal leakage or not, this differs
                if self.no_temporal_leakage:
                    curr_session_position = states[consts.STATE_VARIABLES.SessionPosition.value]
                    curr_session_length = self.df_Y[i][0]
                else:
                    curr_session_position = states[consts.TEMPORAL_LEAKAGE_STATE_VARIABLES.SessionPosition.value]
                    curr_session_length = states[consts.TEMPORAL_LEAKAGE_STATE_VARIABLES.SessionLength.value]

                # append states to our list of states `pr_states`
                pr_states.append(states)

                actions, internals = self.agent.act(
                    states=states, internals=internals,
                    independent=True, deterministic=True
                )

                # if in the second half, store action prediction in array
                if curr_session_position > (curr_session_length / 2):
                    pr_preds.append(actions)

                    # same goes for the current skip. Depending on whether we have temporal leakage or not, this differs
                    if self.no_temporal_leakage:
                        curr_skip = self.df_Y[i][2]
                    else:
                        curr_skip = self.df_Y[i][0]

                    gt_preds.append(curr_skip)

                states, terminal, _ = self.env.execute(actions=actions)
                i += 1

                # if terminal state, append all predictions and groud truths to arrays
                if terminal:
                    pr_vals.append(pr_preds)
                    pr_preds = []
                    gt_vals.append(gt_preds)
                    gt_preds = []

        return pr_vals, gt_vals, pr_states

    def __save_to_txt_file(self, content, filename):
        newfile = open(filename, "w")

        for row in content:
            # convert array of ints to string
            str_row = ""
            for digit in row:
                str_row += str(int(digit))
            newfile.write(str_row + "\n")
        newfile.close()