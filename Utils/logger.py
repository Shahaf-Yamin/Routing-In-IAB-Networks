import datetime
import logging
import os
import torch
# from torch.utils.tensorboard import SummaryWriter

def set_logger(config):
    ''' define logger object to log into file and to stdout '''

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(message)s")

    if not config.quiet:
        consoleHandler = logging.StreamHandler()
        consoleHandler.setFormatter(logFormatter)
        logger.addHandler(consoleHandler)

    log_path = os.path.join(config.result_dir, "logger.log")
    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    logger.propagate = False

def set_logger_and_tracker(config):
    '''
    configure the mlflow tracker:
        1. set tracking location (uri)
        2. configure exp name/id
        3. define parameters to be documented
    '''

    # run_name = config.NETWORK["Environment"]
    config.result_dir = os.path.join('../Results',
                                      config.experiement_name,
                                      config.run_name,
                                      config.ver_name,
                                      config.tag_name,
                                      f'seed_{str(config.seed)}',
                                      "{}".format(datetime.datetime.now().strftime("%Y-%m-%d/%H-%M-%S")))
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)

    set_logger(config)

    train_log_dir = os.path.join(config.result_dir, 'train')
    config.train_writer = None #SummaryWriter(train_log_dir)

    test_log_dir = os.path.join(config.result_dir, 'test')
    config.test_writer = None #SummaryWriter(test_log_dir)

    logger = logging.getLogger("logger")
    logger.info('Result Directory - ' + config.result_dir)
    # Print out the current configurations
    for key, value in config.items():
        if isinstance(value,dict):
            print(key+':')
            for nested_key in value:
                print('\t'+nested_key, ':', value[nested_key])
        else:
            print(key+': '+ str(value))