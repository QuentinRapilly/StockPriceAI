class VitConfig():
    config = {
        "model":{
            "period_size" : 150,
            "patch_size" : 10,
            "nb_pred" : 10,
            "dim" : 60,
            "depth" : 8,
            "heads" : 10,
            "mlp_dim" : 60,
            "pool" : 'cls',
            "channels" : 1,
            "dim_head" : 64,
            "dropout" : 0.1,
            "emb_dropout" : 0.1
            
        },
        
        "dataset_train":{
            "start_date": '1950-01-03',
            "end_date": '2000-01-01',
            "interval_date": '1d',
            "input_size":150,
            "output_size" : 10,
            "batch_size": 256,
            "shuffle":False
        },

        "dataset_test":{
            "start_date": '2000-01-02',
            "end_date": '2010-01-01',
            "interval_date": '1d',
            "input_size":150,
            "output_size" : 10,
            "batch_size": 256,
            "shuffle":False
        },

        "dataset_val":{
            "start_date": '2010-01-02',
            "end_date": '2020-01-01',
            "interval_date": '1d',
            "input_size":150,
            "output_size" : 10,
            "batch_size": 256,
            "shuffle":False
        },

        "learning":{
            "init_lr": 1e-04,
            "lr_decay": 0.99,
            "init_epoch": 5,
            "max_epoch": 100
        } 
    }