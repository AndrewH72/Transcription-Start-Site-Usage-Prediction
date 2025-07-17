import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt

combined_data = [file for file in glob.glob1("data/RNASeq_bw", "*stranded*") if "unstranded" not in file]
combined_tab_files = [file for file in combined_data if file.endswith(".tab")]
combined_tab_files_name_sorted = sorted(combined_tab_files, key=lambda x: x[0])
combined_tab_files_names = [file[ :file.find("_")] for file in combined_tab_files]

# Getting the sum of RNA signals from the TSS onward.
GM12878_files = []
K562_files = []

GM12878_gene_list = pd.read_csv("data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt", sep="\t")
K562_gene_list = pd.read_csv("data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt", sep="\t")
rna_cage = pd.read_csv("data/RNA_CAGE.txt", sep="\t")

with pd.option_context("mode.chained_assignment", None):
    for file in tqdm(combined_tab_files, desc="File"):
        file_name = file[ :file.find("_")]
        cell_line = file[ :file.find(".")]
        gene_list = None

        if cell_line == "K562":
            gene_list = K562_gene_list
        elif cell_line == "GM12878":
            gene_list = GM12878_gene_list

        df = pd.read_csv(f"data/RNASeq_bw/{file}", sep="\t")
        df = df.iloc[:, :-1]

        # Summing from the TSS to the right of the sample.
        df_tss = df.iloc[:, 62:]
        df_tss.loc[:, "ENSID"] = gene_list.loc[:, "Ensembl_ID"]
        df_tss.loc[:, "Sum"] = df_tss.iloc[:, :-1].sum(axis=1)
        reorder = list(df_tss)[-2:-1] + list(df_tss)[:-2] + list(df_tss)[-1:]
        df_tss = df_tss.loc[:, reorder]

        if cell_line == "K562":
            K562_files.append(df_tss.loc[:, "Sum"])
        elif cell_line == "GM12878":
            GM12878_files.append(df_tss.loc[:, "Sum"])

        plt.plot(range(len(df_tss)), df_tss.loc[: ,"Sum"])
        plt.xlabel("Gene")
        plt.ylabel("Sum of Bins")
        plt.title("Genes vs Sum of the TSS and the Bins to the Right")
        plt.savefig(f"plots/dist_tss_sum/{file_name}.sum_tss_bins.png")
        plt.clf()

gm12878 = pd.DataFrame(GM12878_files).T
gm12878.to_csv("./data/gm12878_tss_sum.csv", index=False)
k562 = pd.DataFrame(K562_files).T
k562.to_csv("./data/k562_tss_sum.csv", index=False)

# Our RNA Seq vs Our CAGE-Seq
gm_gene_list = pd.read_csv("data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt", sep="\t")
k5_gene_list = pd.read_csv("data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt", sep="\t")

gm12878 = pd.read_csv("data/gm12878_tss_sum.csv")
gm12878[["Ensembl_ID", "TSS"]] = gm_gene_list[["Ensembl_ID", "tss"]]

k562 = pd.read_csv("data/k562_tss_sum.csv")
k562[["Ensembl_ID", "TSS"]] = k5_gene_list[["Ensembl_ID", "tss"]]

rna_cage = pd.read_csv("data/RNA_CAGE.txt", sep="\t")

gm_cage = gm12878.merge(rna_cage, left_on="Ensembl_ID", right_on="ENSID", how="inner", suffixes=["_gm12878", "_cage"])
gm_cage = gm_cage[["Sum", "Sum.1", "Sum.2", "Sum.3", "GM12878_CAGE_128*3_sum"]]
gm_cage.columns = range(len(gm_cage.columns))
gm_cage = gm_cage.dropna()

k5_cage = k562.merge(rna_cage, left_on="Ensembl_ID", right_on="ENSID", how="inner", suffixes=["_k562", "_cage"])
k5_cage = k5_cage[["Sum", "Sum.1", "Sum.2", "Sum.3", "K562_CAGE_128*3_sum"]]
k5_cage.columns = range(len(k5_cage.columns))
k5_cage = k5_cage.dropna()

gm_cage_correlation = pd.DataFrame()
for col in gm_cage.columns[0:4]:
    file_name = combined_tab_files_name_sorted[int(col)][ :combined_tab_files_name_sorted[int(col)].find("_")]

    pearson_r_values = stats.pearsonr(np.log(gm_cage[4] + 1), np.log(gm_cage[col] + 1))
    gm_cage_correlation = pd.concat([gm_cage_correlation, pd.DataFrame(pearson_r_values).T])

    plt.scatter(gm_cage[4], gm_cage[col])
    plt.xlabel("CAGE Values (Log)")

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("RNA Seq Values (Log)")
    plt.title(f"CAGE vs RNA Seq Values for {file_name}", fontsize=8)
    # plt.show()
    plt.savefig(f"plots/rnaseq_vs_rnacage/{file_name}.rna_seq_xpresso_log.png")
    plt.clf()

k5_cage_correlation = pd.DataFrame()
for col in k5_cage.columns[0:4]:
    file_name = combined_tab_files_name_sorted[int(col) + 4][ :combined_tab_files_name_sorted[int(col) + 4].find("_")]

    pearson_r_values = stats.pearsonr(np.log(k5_cage[4] + 1), np.log(k5_cage[col] + 1))
    k5_cage_correlation = pd.concat([k5_cage_correlation, pd.DataFrame(pearson_r_values).T])

    plt.scatter(k5_cage[4], k5_cage[col])
    plt.xlabel("CAGE Values (Log)")

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("RNA Seq Values (Log)")
    plt.title(f"CAGE vs RNA Seq Values for {file_name}", fontsize=8)
    # plt.show()
    plt.savefig(f"plots/rnaseq_vs_rnacage/{file_name}.rna_seq_xpresso_log.png")
    plt.clf()

gm_cage_correlation.to_csv("csvs/correlation/rnaseq_vs_rnacage/gm_seq_vs_cage.csv", index=False)
k5_cage_correlation.to_csv("csvs/correlation/rnaseq_vs_rnacage/k5_seq_vs_cage.csv", index=False)

# Xpresso CAGE vs Our RNA Seq
rna_cage = pd.read_csv("data/RNA_CAGE.txt", sep="\t")
links_gm12878 = pd.read_csv("data/GM12878_enhancer_gene_links_100kb.hg38.tsv", sep="\t")
xpresso = pd.read_csv("data/GM12878_K562_18377_gene_expr_fromXpresso.csv")

xpresso["chrom"] = "chr" + xpresso["chrom"]
rna_cage_links_gm12878 = rna_cage.merge(links_gm12878, left_on="ENSID", right_on="TargetGeneEnsembl_ID", how="inner", suffixes=["_rna_cage", "_links"])
rna_cage_links_gm12878_xpresso = rna_cage_links_gm12878.merge(xpresso, left_on="ENSID_old", right_on="gene_id", how="inner", suffixes=["_rna_cage", "_xpresso"])
rna_cage_links_gm12878_xpresso["diffTSS-dist"] = abs(rna_cage_links_gm12878_xpresso["TargetGeneTSS"] - rna_cage_links_gm12878_xpresso["TSS_xpresso"])
min_dist_indx = rna_cage_links_gm12878_xpresso.groupby("ENSID_old")["diffTSS-dist"].idxmin()

xpresso_cage_gm12878 = rna_cage_links_gm12878_xpresso.loc[min_dist_indx][["ENSID_old", "GM12878_CAGE_128*3_sum_xpresso"]]
xpresso_cage_k562 = rna_cage_links_gm12878_xpresso.loc[min_dist_indx][["ENSID_old", "K562_CAGE_128*3_sum_xpresso"]]

gm12878 = pd.read_csv("./data/gm12878_tss_sum.csv")
k562 = pd.read_csv("./data/k562_tss_sum.csv")

gm12878.columns = range(len(gm12878.columns))
gm12878.insert(0, "Ensembl_ID", GM12878_gene_list["Ensembl_ID"])
gm12878.insert(1, "TSS_gm12878", GM12878_gene_list["tss"])

k562.columns = range(len(k562.columns))
k562.insert(0, "Ensembl_ID", K562_gene_list["Ensembl_ID"])
k562.insert(1, "TSS_k562", K562_gene_list["tss"])

gm_xpresso = gm12878.merge(xpresso_cage_gm12878, left_on="Ensembl_ID", right_on="ENSID_old", how="inner", suffixes=["_gm12878", "_xpresso"])
gm_xpresso = gm_xpresso[[0, 1, 2, 3, "GM12878_CAGE_128*3_sum_xpresso"]]
k5_xpresso = k562.merge(xpresso_cage_k562, left_on="Ensembl_ID", right_on="ENSID_old", how="inner", suffixes=["_gm12878", "_xpresso"])
k5_xpresso = k5_xpresso[[0, 1, 2, 3, "K562_CAGE_128*3_sum_xpresso"]]
gm_xpresso_correlation = pd.DataFrame()
for col in gm_xpresso.columns[0:4]:
    file_name = combined_tab_files_name_sorted[int(col) - 1][ :combined_tab_files_name_sorted[int(col) - 1].find("_")]

    pearson_r_values = stats.pearsonr(np.log(gm_xpresso["GM12878_CAGE_128*3_sum_xpresso"] + 1), np.log(gm_xpresso[col] + 1))
    gm_xpresso_correlation = pd.concat([gm_xpresso_correlation, pd.DataFrame(pearson_r_values).T])

    plt.scatter(gm_xpresso["GM12878_CAGE_128*3_sum_xpresso"], gm_xpresso[col])
    plt.xlabel("Xpresso Values (Log)")

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("RNA Seq Values (Log)")
    plt.title("GM12878 Xpresso vs RNA Seq Values")
    plt.savefig(f"plots/rnaseq_vs_xpresso/{file_name}.rna_seq_xpresso_inner_log.png")
    plt.clf()

k5_xpresso_correlation = pd.DataFrame()
for col in k5_xpresso.columns[0:4]:
    file_name = combined_tab_files_name_sorted[int(col) - 1 + 4][ :combined_tab_files_name_sorted[int(col) - 1 + 4].find("_")]

    pearson_r_values = stats.pearsonr(np.log(k5_xpresso["K562_CAGE_128*3_sum_xpresso"] + 1), np.log(k5_xpresso[col] + 1))
    k5_xpresso_correlation = pd.concat([k5_xpresso_correlation, pd.DataFrame(pearson_r_values).T])

    plt.scatter(k5_xpresso["K562_CAGE_128*3_sum_xpresso"], k5_xpresso[col])
    plt.xlabel("Xpresso Values (Log)")

    plt.xscale("log")
    plt.yscale("log")

    plt.ylabel("RNA Seq Values (Log)")
    plt.title("K562 Xpresso vs RNA Seq Values")
    plt.savefig(f"plots/rnaseq_vs_xpresso/{file_name}.rna_seq_xpresso_inner_log.png")
    plt.clf()

gm_xpresso_correlation.to_csv("csvs/correlation/rnaseq_vs_xpresso/gm12878_rnaseq_vs_xpresso.csv", index=False)
k5_xpresso_correlation.to_csv("csvs/correlation/rnaseq_vs_xpresso/k562_rnaseq_vs_xpresso.csv", index=False)