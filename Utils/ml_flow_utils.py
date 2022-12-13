import argparse
import logging
import os
import glob
import shutil
import sys
import torch
from .config import read_config
from .logger import set_logger_and_tracker
import subprocess
import wandb

logger = logging.getLogger("logger")


def is_git_directory(path='.'):
    return subprocess.call(['git', '-C', path, 'status'], stderr=subprocess.STDOUT, stdout=open(os.devnull, 'w')) == 0

def get_args():
    """" collects command line arguments """

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config',                  default='../Setting.json', type=str, help='configuration file')
    argparser.add_argument('--exp_name',                default=None, type=str, help='experiment name')
    argparser.add_argument('--run_name',                default=None, type=str, help='run name')
    argparser.add_argument('--ver_name',                default=None, type=str, help='ver name')
    argparser.add_argument('--tag_name',                default=None, type=str, help='tag name')
    argparser.add_argument('--batch_size',              default=None, type=int, help='batch size in training')
    argparser.add_argument('--seed',                    default=None, type=int, help='randomization seed')
    argparser.add_argument('--gpu',                     default=None, type=int, help='gpu index we would like to run on')
    argparser.add_argument('--ue_speed',                default=None, type=int, help='users movement speed in meter/sec.')
    argparser.add_argument('--lr_freq',                 default=None, type=int, help='This frequncy dicatates how frequt our agent will learn to update his policy from experience.')
    argparser.add_argument('--quiet',                   dest='quiet', action='store_true')
    argparser.set_defaults(quiet=False)
    argparser.add_argument('--epsilon', type=float, default=None)
    argparser.add_argument('--lr', type=float, default=None)
    argparser.add_argument('--buffer_size', type=int, default=None)
    argparser.add_argument('--pre_trained_path', type=str, default=None)
    argparser.add_argument('--algorithm', type=int, default=None, choices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], help="Supported Algorithms: 0. Q-Routing, 1. Shortest-Path, 2. Back-Pressure, 3. Full Echo Q-Routing, 4. Random Routing, 5. Tabular-Actor-Critic, 6. Deep-Actor-Critic 7. Relational-Actor-Critic, 8. Decentralized-Relational-Actor-Critic, 9.Federated-Relational-Actor-Critic")
    argparser.add_argument('--mode', type=int, default=0, choices=[0, 1, 2, 3], help="Supported Algorithms: 0. Different Loads (Train+Test), 1. Online Changing Load Test Only, 2. Changing Topology Test Only, 3. Online Node Failure Test Only")
    args = argparser.parse_args()
    return args

def gpu_init(args):
    """ Allows GPU memory growth """

    if args.gpu is None:
        device = torch.device('cpu')
    else:
        device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    print('Using device:', device)
    logger.info(f'Using device:{device}')
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        logger.info(f'Device Name:{torch.cuda.get_device_name(0)}')
        print('Memory Usage:')
        logger.info('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1), 'GB')
        logger.info(f'Allocated:{round(torch.cuda.memory_allocated(0) / 1024 ** 3, 1)}GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1), 'GB')
        logger.info(f'Cached: {round(torch.cuda.memory_reserved(0) / 1024 ** 3, 1)}GB')

    return device

def save_scripts(config):
    path = os.path.join(config.result_dir, 'scripts')
    if not os.path.exists(path):
        os.makedirs(path)

    scripts_to_save = glob.glob('../**/*.py', recursive=True) + [config.config]
    scripts_to_save = [script for script in scripts_to_save if 'venv' not in script and 'Results' not in script]
    if scripts_to_save is not None:
        for script in scripts_to_save:
            if 'Results' not in script:
                dst_file = os.path.join(path, os.path.basename(script))
                shutil.copyfile(os.path.join(os.getcwd(), script), dst_file)

def generate_result_directory(config):
    def generate_result_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)
        if not os.path.exists(path + 'plots') or not os.path.exists(path + 'data'):
            os.makedirs(path + 'plots')
            os.makedirs(path + 'data')
    if is_git_directory():
        gitHash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode("utf-8")
        print(gitHash)
        logger.info(gitHash)
    train_path = config.result_dir + '/train/'
    test_path = config.result_dir + '/test/'
    generate_result_dir(train_path)
    generate_result_dir(test_path)
    return

def init_wandb(config):
    name = config["experiement_name"] + '_' + config["run_name"] + '_' + config["ver_name"]
    experiment = wandb.init(project="thesis_project", entity="shahaf_yamin", resume='allow', name=name, config=config)
    return experiment, experiment.config

def preprocess_meta_data():
    """ preprocess the config for specific run:
            1. reads command line arguments
            2. updates the config file and set gpu config
            3. configure gpu settings
            4. Define logger
            5. Generate Result Directory
            6. Save scripts
    """

    args = get_args()

    config = read_config(args)

    experiment = None

    # experiment, config = init_wandb(config)

    device = gpu_init(args)

    set_logger_and_tracker(config)

    generate_result_directory(config)

    save_scripts(config)

    return config, args, device, experiment