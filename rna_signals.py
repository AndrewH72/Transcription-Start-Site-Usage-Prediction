import sys
sys.path.append("/home/coder/diffTSS")

import os
import kipoiseq
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
from scripts.utils import FastaStringExtractor, one_hot_encode

data = os.listdir("/home/coder/data/RNASeq_bw")
coverage_files = sorted([file for file in data if file.endswith(".txt")], key=lambda x: (x[x.find(".") + 1], x[0]))

# Generating the lineplot of the average signal value across the target interval from each coverage file.
enhancer_gene_K562_100kb = pd.read_csv("./data/K562_enhancer_gene_links_100kb.hg38.tsv", sep="\t")
enhancer_gene_GM12878_100kb = pd.read_csv("./data/GM12878_enhancer_gene_links_100kb.hg38.tsv", sep="\t")

gene_K562_tss = pd.read_csv("./data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt", sep="\t")[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
gene_GM12878_tss = pd.read_csv("./data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt", sep="\t")[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]

for file in tqdm(coverage_files, desc="File"):
    cell_line = file[ :file.find(".")]
    file_name = file[ :file.find("E") + 11]
    enhancer_gene_100kb = None
    gene_tss = None

    if cell_line == "K562":
        enhancer_gene_100kb = enhancer_gene_K562_100kb
        gene_tss = gene_K562_tss
    elif cell_line == "GM12878":
        enhancer_gene_100kb = enhancer_gene_GM12878_100kb
        gene_tss = gene_GM12878_tss

    rna_df = pd.read_csv(f"./data/RNASeq_bw/{file}", header=None, sep="\t")
    rna_df.columns = range(len(rna_df.columns))

    plus_rna_df = rna_df[rna_df[5] == "+"]
    minus_rna_df = rna_df[rna_df[5] == "-"]
    
    gene_tss["ENSID"] = gene_tss["Ensembl_ID"]
    enhancer_gene_100kb_includeNoEnhancerGene = enhancer_gene_100kb.merge(gene_tss, left_on="TargetGeneEnsembl_ID", right_on="Ensembl_ID", how="right", suffixes=["", "_gene"]).reset_index()
    gene_list = list(gene_tss["ENSID"])

    gene_enhancer_table = enhancer_gene_100kb_includeNoEnhancerGene
    rna_method = "encoding"

    rna_df_list = []
    plus_rna_df_list = []
    minus_rna_df_list = []

    for gene in tqdm(gene_list, desc="Gene"):
        gene_df = gene_enhancer_table[gene_enhancer_table["ENSID"] == gene]
        gene_enhancer_df = gene_df

        if rna_method is not None:
            if rna_method == "encoding" or rna_method == "one-hot":
                gene_rna_df = rna_df[rna_df[3] == gene]
                plus_gene = plus_rna_df[plus_rna_df[3] == gene]
                minus_gene = minus_rna_df[minus_rna_df[3] == gene]
            if rna_method == "embedding":
                gene_rna_df = rna_df[gene_list.index(gene)]
                plus_gene = plus_rna_df[gene_list.index(gene)]
                minus_gene = minus_rna_df[gene_list.index(gene)]

            # Grabs the DNA sequence.
            fasta_path = "./data/hg38.fa"
            fasta_extractor = FastaStringExtractor(fasta_path)

            # Promoter will be the first row, then enhancers will be sorted by distance from promoter (increasing).
            gene_pe = gene_enhancer_df.sort_values(by="distance")
            row_0 = gene_pe.iloc[0]
            gene_ensid = row_0["TargetGeneEnsembl_ID"]
            gene_name = row_0["TargetGene"]
            gene_tss = row_0["TargetGeneTSS"]
            chrom = row_0["chr"]

            if row_0["TargetGeneTSS"] != row_0["TargetGeneTSS"]:
                gene_tss = row_0["tss"]
                gene_name = row_0["name_gene"]
                chrom = row_0["chr_gene"]

            # Grabs the DNA sequence of our TSS interval.
            max_seq_len = 2000
            target_interval = kipoiseq.Interval(chrom, int(gene_tss - max_seq_len / 2), int(gene_tss + max_seq_len / 2))
            promoter_seq = fasta_extractor.extract(target_interval)
            promoter_code = one_hot_encode(promoter_seq)

            # Grabs the RNA signals from each enhancer.
            if rna_method == "encoding" or rna_method == "one-hot":
                gene_len = row_0["end"] - row_0["start"]
                rna_signal = gene_rna_df[[9]]
                gene_rna_df = gene_rna_df[(gene_rna_df[7] >= target_interval.start) & (gene_rna_df[8] <= target_interval.end)]
                new_index = gene_rna_df[7].values - target_interval.start
                rna_signal = rna_signal.set_index(new_index).reindex(list(range(0, max_seq_len)), fill_value=0)
                gene_rna_df = np.array(rna_signal).flatten()

                plus_rna_signal = plus_gene[[9]]
                plus_gene = plus_gene[(plus_gene[7] >= target_interval.start) & (plus_gene[8] <= target_interval.end)]
                new_index = plus_gene[7].values - target_interval.start
                plus_rna_signal = plus_rna_signal.set_index(new_index).reindex(list(range(0, max_seq_len)), fill_value=0)
                plus_gene = np.array(plus_rna_signal).flatten()

                minus_rna_signal = minus_gene[[9]]
                minus_gene = minus_gene[(minus_gene[7] >= target_interval.start) & (minus_gene[8] <= target_interval.end)]
                new_index = minus_gene[7].values - target_interval.start
                minus_rna_signal = minus_rna_signal.set_index(new_index).reindex(list(range(0, max_seq_len)), fill_value=0)
                minus_gene = np.array(minus_rna_signal).flatten()
            if rna_method == "embedding":
                gene_rna_df = np.concatenate([gene_rna_df.reshape(1, 125), np.zeros([60, 125])])

                plus_gene = np.concatenate([plus_gene.reshape(1, 125), np.zeros([60, 125])])

                minus_gene = np.concatenate([minus_gene.reshape(1, 125), np.zeros([60, 125])])

            rna_df_list.append(gene_rna_df)
            plus_rna_df_list.append(plus_gene)
            minus_rna_df_list.append(minus_gene)

    df = pd.DataFrame(rna_df_list)
    plus_df = pd.DataFrame(plus_rna_df_list)
    minus_df = pd.DataFrame(minus_rna_df_list)

    df.to_csv("./REU/RNA-Signal_csv/{file_name}.csv")
    plus_df.to_csv(f"./REU/Plus-Minus-RNA_csv/{file_name}.plus.csv")
    minus_df.to_csv(f"./REU/Plus-Minus-RNA_csv/{file_name}.minus.csv")

    # Plotting the average RNA signal for the entire file.
    plt.plot(df.columns, df.mean())
    plt.xticks(range(0, 2000, 100))
    plt.xlabel("Position in 2Kbp Promoter Window")
    plt.ylabel("Averaged RNA Signal")
    plt.title(f"Distribution of Averaged RNA Signals for {file_name}")
    plt.savefig(f"./REU/plots/{file_name}.average_rna_signal.png")
    plt.clf()

    # # Plotting just the plus dataframe.
    # plt.plot(plus_df.columns, plus_df.mean())
    # plt.xticks(range(0, 2000, 200))
    # plt.xlabel("position in 2kbp promoter window")
    # plt.ylabel("averaged rna signal")
    # plt.title(f"distribution of averaged rna signals for gm12878.minus.encff074sxq.plus", fontsize=8)
    # plt.savefig(f"./reu/plots/{file_name}.plus.average_rna_signals.png")
    # plt.clf()

    # # Plotting just the minus dataframe.
    # plt.plot(minus_df.column, minus_df.mean())
    # plt.xticks(range(0, 2000, 200))
    # plt.xlabel("position in 2kbp promoter window")
    # plt.ylabel("averaged rna signal")
    # plt.title(f"distribution of averaged rna signals for gm12878.minus.encff074sxq.minus", fontsize=8)
    # plt.savefig(f"./reu/plots/{file_name}.minus.average_rna_signals.png")
    # plt.clf()

    # Plotting both dataframes against each other.
    plt.plot(plus_df.columns, plus_df.mean())
    plt.plot(minus_df.columns, minus_df.mean())
    plt.xticks(range(0, 2000, 200))
    plt.xlabel("Position in 2kbp Promoter Window")
    plt.ylabel("Averaged Rna Signal")
    plt.title(f"Distribution of Averaged RNA Signals for {file_name} Plus and Minus Strands", fontsize=8)
    plt.savefig(f"./REU/plots/{file_name}.plus_minus.average_rna_signals.png")
    plt.clf()