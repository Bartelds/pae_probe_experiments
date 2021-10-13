#!/usr/bin/env python
import os
import pandas as pd
import seaborn as sns

from argparse import ArgumentParser
from glob import glob
from pathlib import Path


def process_data(outputs, models):
    """Reads stdout and returns it as a DataFrame"""

    scores = []
    ft_method = []
    clfs = []
    columns = []
    for output in outputs:
        clf = output.split("_")[-1][:-7]
        with open(output, "r") as f:
            lines = f.readlines()
            for line in lines:
                if line.strip()[0] in ["t", "0"] or line.strip().split()[0] in models:
                    lsts = line.split()
                    if lsts[0] == "0":
                        scores.append(lsts[1:])
                    if lsts[0] in models:
                        ft_method.append(lsts)
                    if lsts[0] =="train":
                        columns.append(lsts)
                        clfs.append(clf)
    
    ft_model = [sublist[0] for sublist in ft_method]
    ft_layer = ["-".join(sublist[1:]) for sublist in ft_method]
    df = pd.DataFrame(scores, columns=columns[0])
    df["method"] = ft_model
    df["layer"] = ft_layer
    df["classifier"] = clfs
    df[["acc", "precision", "recall", "f1"]] = df[["acc", "precision", "recall", "f1"]].apply(pd.to_numeric)

    return df


def plot_scores(df, output_dir, eval):
    """Plots phone classification performance for each model in DataFrame. Plots F1 score by default"""

    output_dir = Path(output_dir, eval)
    print(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_group = df.groupby("method")
    for index, group in df_group:
        print(f"Plotting results for {index}")
        f = sns.lineplot(data=group, x="layer", y="f1", hue="classifier").set_title(f"Phone-classification {eval} scores for {index} on TIMIT")
        f.figure.savefig(str(output_dir) + f"/{index}_{eval}.png")
        f.figure.clf()
    print(f"Plots stored at {output_dir}")

def main():
    parser = ArgumentParser(
        prog="W2v2 plot phone classification",
        description="Creates phone classification plots for different w2v2 models",
    )
    parser.add_argument("-e", "--eval", default="f1", choices=["acc", "precision", "recall", "f1"])
    parser.add_argument("-o", "--output_dir", default="../experiments/wav2vec2.0/plots/")
    args = parser.parse_args()

    outputs = glob("../experiments/wav2vec2.0/logs/*.stdout")
    models = ["wav2vec2-large-960h", "wav2vec2-large-xlsr-53"]

    df = process_data(outputs, models)
    plot_scores(df, args.output_dir, args.eval)

if __name__ == "__main__":
    main()
