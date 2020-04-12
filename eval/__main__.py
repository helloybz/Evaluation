import argparse
import glob
import os
import pickle
import sys

import pandas as pd

from .experiments import NodeClassification


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', type=str)
    parser.add_argument('--input_path', type=str)
    # parser.add_argument('--output_path', type=str)
    config = parser.parse_args()

    process(config)

def process(config):
    # load inputs
    z_paths = glob.glob(os.path.join(config.input_path,"z_*.pt"))
    label_path = os.path.join(config.input_path, 'y.pt')

    # load model
    if config.experiment == 'clf':
        model_class = NodeClassification

    macro_f1_df = pd.DataFrame(
        data=0,
        index=[os.path.basename(path) for path in z_paths],
        columns = [i/10 for i in range(1, 10)]
    )
    micro_f1_df = pd.DataFrame(
        data=0,
        index=[os.path.basename(path) for path in z_paths],
        columns = [i/10 for i in range(1, 10)]
    )
    for z_path in z_paths:
        x = pickle.load(open(z_path, 'rb'))
        y = pickle.load(open(label_path, 'rb'))
        for train_ratio in [i/10 for i in range(1, 10)]:
            for idx in range(10):
                model = model_class(x, y, train_ratio=train_ratio)
                model.train()
                macro, micro = model.test()
                macro_f1_df.loc[os.path.basename(z_path), train_ratio] += macro / 10
                micro_f1_df.loc[os.path.basename(z_path), train_ratio] += micro / 10

    micro_f1_df.to_csv(
        os.path.join(os.path.dirname(config.input_path), f"{config.experiment}_micro_result.csv")
    )
    macro_f1_df.to_csv(
        os.path.join(os.path.dirname(config.input_path), f"{config.experiment}_macro_result.csv")
    )

if __name__ == "__main__":
    sys.exit(main())