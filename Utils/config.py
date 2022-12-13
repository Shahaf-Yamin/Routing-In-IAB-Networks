from bunch import Bunch
from collections import OrderedDict
import json
import logging
from pathlib import Path
from random import randint

CONFIG_VERBOSE_WAIVER = ['save_model', 'tracking_uri', 'quiet', 'sim_dir', 'train_writer', 'test_writer', 'valid_writer']
MAX_SEED = 1000000
logger = logging.getLogger("logger")

algorithm_mapping = {0: 'Q-Routing',
                     1: "Shortest-Path",
                     2: "Back-Pressure",
                     3: "Full-Echo-Q-Routing",
                     4: 'Random',
                     5: 'Tabular-Actor-Critic',
                     6: 'Deep-Actor-Critic',
                     7:'Relational-Actor-Critic',
                     8:'Decentralized-Relational-Actor-Critic',
                     9:'Federated-Relational-Actor-Critic'
                     }

class Config(Bunch):
    """ class for handling dicrionary as class attributes """

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def print(self):
        def print_nested_dict(_key, _val):
            logger.info("| {:35s} | ( |\n".format(_key) + line)
            for nested_key, nested_val in sorted(_val.items(), key=lambda x: x[0]):
                if isinstance(nested_val, OrderedDict):
                    print_nested_dict(nested_key, nested_val)
                else:
                    if nested_key not in CONFIG_VERBOSE_WAIVER:
                        logger.info("| {:35s} | {:80} |\n".format(nested_key, str(nested_val)) + line)
            logger.info("| {:35s} | ) |\n".format('') + line)
        line_len = 122
        line = "-" * line_len
        logger.info(line + "\n" +
              "| {:^35s} | {:^80} |\n".format('Feature', 'Value') +
              "=" * line_len)
        for key, val in sorted(self.items(), key= lambda x: x[0]):
            if isinstance(val, OrderedDict):
                print_nested_dict(key, val)
            else:
                if key not in CONFIG_VERBOSE_WAIVER:
                    logger.info("| {:35s} | {:80} |\n".format(key, str(val)) + line)
        logger.info("\n")

def read_json_to_dict(fname):
    """ read json config file into ordered-dict """
    fname = Path(fname)
    with fname.open('rt') as handle:
        config_dict = json.load(handle, object_hook=OrderedDict)
        return config_dict

def read_config(args):
    """ read config from json file and update by the command line arguments """
    if args.config is not None:
        json_file = args.config
    else:
        raise ValueError("preprocess config: config path wasn't specified")

    config_dict = read_json_to_dict(json_file)
    config = Config(config_dict)

    for arg in sorted(vars(args)):
        key = arg
        val = getattr(args, arg)
        if val is not None:
            setattr(config, key, val)
    if args.epsilon is not None:
        config["AGENT"]["epsilon"] = args.epsilon
    if args.lr is not None:
        config["DQN"]["optimizer_learning_rate"] = args.lr
    if args.buffer_size is not None:
        config["DQN"]["memory_bank_size"] = args.buffer_size

    if args.seed is None and config.seed is None:
        config.seed = randint(0, MAX_SEED)

    if args.algorithm is not None:
        config["NETWORK"]["Environment"] = algorithm_mapping[args.algorithm]
    if args.pre_trained_path is not None:
        config["AGENT"]["pretrained_path"] = args.pre_trained_path
    if args.ue_speed is not None:
        config["NETWORK"]["ue_speed"] = args.ue_speed
    if args.run_name is not None:
        config.run_name = args.run_name
    if args.ver_name is not None:
        config.ver_name = args.ver_name
    if args.mode == 0:
        config.tag_name = 'e2e'
    elif args.mode == 1:
        config.tag_name = 'onlineChangingLoad'
    elif args.mode == 2:
        config.tag_name = 'TopologyChanges'
    elif args.mode == 3:
        config.tag_name = 'NodeFailure'
    else:
        raise Exception('Invalid configuration')

    return config
