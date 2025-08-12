import json

def write_logs(log_stats: str, filename, config):
    if config.PREPROCESSING_SPLIT != 1:
        with open(filename + "/exp_log_split.txt", 'a+') as f:
            f.write(json.dumps(log_stats) + "\n")

    elif config.PREPROCESSING_QUAZIPOLAR:
        with open(filename + "/exp_log_quazipolar.txt", 'a+') as f:
            f.write(json.dumps(log_stats) + "\n")

    else:
        with open(filename + "/exp_log_single.txt", 'a+') as f:
            f.write(json.dumps(log_stats) + "\n")

def write_single_log(log_stats: str, filename, config):
    with open(filename + "/recall_total.txt", 'a+') as f:
        f.write(json.dumps(log_stats) + "\n")