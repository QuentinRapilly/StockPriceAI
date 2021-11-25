class StockAIConfig():
    config = {
        "model":{
            "input_size": 1,
            "lstm_size": 128,
            "num_layers": 1,
            "keep_prob": 0.8
        },

        "dataset":{
            "start_date": '1950-01-03',
            "end_date": '2021-11-16',
            "interval_date": '1d',
            "nb_samples":20,
            "batch_size": 64,
            "shuffle":False
        },

        "learning":{
            "num_steps": 30,
            "init_lr": 1e-03,
            "lr_decay": 0.99,
            "init_epoch": 5,
            "max_epoch": 30
        }   
    }