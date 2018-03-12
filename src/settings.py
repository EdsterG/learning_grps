from console_utils import mkdir_p
import os.path as osp
PROJECT_PATH = osp.abspath(osp.join(osp.dirname(__file__), '..'))

# Folder containing pddl domains
DOMAIN_DIR = mkdir_p(PROJECT_PATH + "/data/domains/pddl")

# Temp folder location
TEMP_DIR = mkdir_p(PROJECT_PATH + "/data/temp")

# Location to store generated data
DATASET_DIR = mkdir_p(PROJECT_PATH + "/data/dataset")

# Location to store logs
LOG_DIR =  mkdir_p(PROJECT_PATH + "/data/log")
