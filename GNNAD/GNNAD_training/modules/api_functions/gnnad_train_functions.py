import logging
import os
import datetime 

def get_volume_dir(usr_id, inv_id, runid, volume_name, TODAY_DATE):
    """
    returns: str ->  /app/date/user_id/inverter_id/train/run_id/
    that's where individual run data will be stored. 
    """
    train_folder = "train"
    volume_dir = os.path.join(volume_name, TODAY_DATE, str(usr_id), str(inv_id), train_folder, str(runid))
    return volume_dir


def create_folders(model_dir):
    """
    within folders for each run, there will be folders for weights, pickles and mlflow related logs
    model_dir = /app/date/user_id/inverter_id/train/run_id/
    """
    run_folders = ["weights", "pickles", "mlflow_training"]
    for folder in run_folders:
        os.makedirs(model_dir + "/" + folder, exist_ok=True)
    
    mlflow_training_folders = ["input_dataset"]
    for folder in mlflow_training_folders:
        os.makedirs(os.path.join(model_dir, "mlflow_training", folder), exist_ok=True)
    

def setup_logging(log_file=None):
    if log_file is None:
        log_file = f"training_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=log_file,
        filemode='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

    return logging.getLogger(__name__)