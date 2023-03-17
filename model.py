from lib.base_model import BaseModel
from lib.cli_parser import parse_args

class Model(BaseModel):
    def execute(self):
        # dictionary with the model parameters
        model_params = {
            "memory": 10000, "batch_size": 256, "update_frequency": 256, "start_updating": 1000,
            "learning_rate": 0.001, "discount": 0.9,
            "predict_terminal_values": False, "target_sync_frequency": 1000,
            "config": {
                "buffer_observe": False
            }
        }

        # add n-step learning (`horizon`) if present
        if self.n_step > 1:
            model_params["horizon"] = self.n_step

        # add network configuration (MLP or RNN)
        if (self.network_type == "mlp") or (self.network_type == "mlp_obs_stacking"):
            model_params["network"] = { "type": "auto", "size": 128, "depth": 3 }
        elif self.network_type == "lstm":
            model_params["network"] = { "type": "auto", "size": 128, "depth": 2, "rnn": 4 }
        elif self.network_type == "gru":
            model_params["network"] = [
                dict(type="dense", size=128),
                dict(type="dense", size=128),
                dict(type="gru", size=128, horizon=4)
            ]

        # perform training
        self.train(**model_params)

        # perform evaluation
        self.evaluate(shap_analysis=False)

if __name__ == "__main__":
    args = parse_args()

    # run the model 5 times
    for x in range(5):
        Model(args.n_episodes, args.model_name, x + 1, args.dqn_type, args.network, args.n_step, args.no_temporal_leakage, args.test_set).execute()
