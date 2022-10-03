import pandas as pd

from codebase.utils.flops_counter import profile
from evaluator import main as evaluate


import matplotlib.pyplot as plt
import os
import json
import numpy as np
from pymoo.util.nds.fast_non_dominated_sort import fast_non_dominated_sort
import plotly.express as px


def test_configs(dataset):
    list_configs = os.listdir(f"subnets/{dataset}/iter_30")
    list_configs = [i.replace(".config", "") for i in list_configs if "config" in i]
    config_info = {}

    for config in list_configs:
        config_info[config] = {}
        config_info[config]["info"] = json.load(open(f"subnets/{dataset}/iter_30/{config}.stats"))
        config_info[config]["encoding"] = json.load(open(f"subnets/{dataset}/iter_30/{config}.config"))

    for n in [1, 5]:
        flops, top_n, configs_n = [], [], []
        for config in list_configs:
            flops.append(config_info[config]["info"]["flops"])
            top_n.append(config_info[config]["info"][f"top{n}"])
            configs_n.append(config_info[config]["encoding"])

        f_topn = np.column_stack((flops, top_n))
        f_topn[:, 1] = -f_topn[:, 1]
        ndf = fast_non_dominated_sort(f_topn)[0]
        f_topn[:, 1] = -f_topn[:, 1]

        f_topn = f_topn[ndf, :]
        sorted_idx = np.argsort(f_topn[:, 0])
        f_topn = f_topn[sorted_idx, :]
        ndf = [ndf[i] for i in sorted_idx]
        configs_n = [configs_n[i] for i in ndf]

        df = pd.DataFrame(f_topn, columns=["flops", f"top{n} accuracy"])
        df["configs"] = configs_n
        df["configs"] = df["configs"].astype(str)

        fig = px.scatter(df, x="flops", y=f"top{n} accuracy", hover_data=["configs"],
                         title=f"NAT results for {dataset}")
        fig.update_traces(marker=dict(size=12, line=dict(width=2, color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.write_html(f"population/{dataset}/nat_results_top{n}.html")
        # fig.show()

        fig = plt.figure(figsize=(8, 5))
        plt.scatter(f_topn[:, 0], f_topn[:, 1], marker="s", facecolors='none', edgecolors="red")
        plt.plot(f_topn[:, 0], f_topn[:, 1], c="red")
        plt.grid()
        plt.xlabel("flops")
        plt.ylabel(f"top_{n} accuracy")
        plt.title(f"NAT results for {dataset}")
        plt.savefig(f"population/{dataset}/nat_results_top{n}.jpg", dpi=1200)
        # plt.show()


def main():
    datasets = ["cifar10", "cifar100"]
    for dataset in datasets:
        test_configs(dataset)


if __name__ == "__main__":
    main()



