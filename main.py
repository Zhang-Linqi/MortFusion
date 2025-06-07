# -*- coding: utf-8 -*-


from configs import config
from utils.data_preparation import prepare_inputs
from train.model_runner import build_model, train_model, run_history_prediction, run_recursive_prediction

def main():
    data_dict = prepare_inputs(config)
    device = config.DEVICE
    model = build_model(config, device)
    train_model(config, model, data_dict, epochs=config.epochs, lr=config.lr)
    run_history_prediction(config, model, data_dict)
    run_recursive_prediction(config, model, data_dict)

if __name__ == "__main__":
    main()
   