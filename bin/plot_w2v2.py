#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import seaborn as sns

from argparse import ArgumentParser
from glob import glob
from pathlib import Path


def process_data(outputs, models, eval, save):
    """Reads stdout and returns it as a DataFrame"""

    scores = []
    ft_method = []
    clfs = []
    columns = []
    corpus = []
    for output in outputs:
        clf = output.split("_")[-1][:-7]
        with open(output, "r") as f:
            lines = f.readlines()
            for line in lines:
                lsts = line.strip().split()
                if len(lsts) > 0:
                    if lsts[0] == "0":
                        scores.append(lsts[1:])
                        if lsts[1] == lsts[2]:
                            corpus.append(lsts[1])
                    if lsts[0] in models:
                        ft_method.append(lsts)
                    if lsts[0] =="train":
                        columns.append(lsts)
                        clfs.append(clf)

    ft_model = [sublist[0] for sublist in ft_method]
    ft_layer = ["-".join(sublist[1:]) for sublist in ft_method]
    scores = [score for score in scores if score[0] == lsts[1]]
    df = pd.DataFrame(scores, columns=columns[0])
    df["method"] = ft_model
    df["layer"] = ft_layer
    df["classifier"] = clfs
    df[["acc", "precision", "recall", "f1"]] = df[["acc", "precision", "recall", "f1"]].apply(pd.to_numeric)
    
    if save:
        save_dir = "../experiments/wav2vec2.0/raw/" + str(list(set(corpus))[0]) + "/"
        os.makedirs(save_dir, exist_ok=True)
        df.to_csv(save_dir + eval + "_" + "_".join(models) + ".csv", index=False)

    return df, str(list(set(corpus))[0])


def plot_scores(df, corpus, output_dir, eval):
    """Plots phone classification performance for each model in DataFrame. Plots F1 score by default"""

    output_dir = Path(output_dir, corpus, eval)
    os.makedirs(output_dir, exist_ok=True)
    
    df_group = df.groupby("method")
    for index, group in df_group:
        print(f"Plotting results for {index}")
        f = sns.lineplot(data=group, x="layer", y="f1", hue="classifier", ci=None).set_title(f"Phone-classification {eval} scores for {index} on {group['train'].to_list()[0]}")
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
    parser.add_argument("-s", "--save", default=False)
    args = parser.parse_args()

    outputs = glob("../experiments/wav2vec2.0/logs/*.stdout")
    models = ["wav2vec2-large", "wav2vec2-large-960h", "wav2vec2-large-xlsr-53", "wav2vec2-xlsr-53-phon-cv-babel-ft"]
    
    df, corpus = process_data(outputs, models, args.eval, args.save)
    plot_scores(df, corpus, args.output_dir, args.eval)

if __name__ == "__main__":
    main()
