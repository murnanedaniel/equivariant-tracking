import sys
import argparse
import yaml
import time

import warnings
warnings.filterwarnings('ignore')

sys.path.append("../")
from train_lightning import train, test



def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("train_gnn.py")
    add_arg = parser.add_argument
    add_arg("config", nargs="?", default="default_config.yaml")
    add_arg("root_dir", nargs="?", default=None)
    add_arg("checkpoint", nargs="?", default=None)
    add_arg("random_seed", nargs="?", default=None)
    return parser.parse_args()

def main():
    print(time.ctime())

    args = parse_args()

    with open(args.config) as file:
        print(f"Using config file: {args.config}")
        default_configs = yaml.load(file, Loader=yaml.FullLoader)

    num_runs = default_configs["num_runs"]

    for i in range(num_runs):
        print(f"Run {i+1}/{num_runs}")
        model, trainer = train(default_configs, args.root_dir, args.checkpoint, args.random_seed)

        print("Running test")
        test_results = test(model, trainer)
    
    
if __name__ == "__main__":

    main()
