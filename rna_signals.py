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
enhancer_gene_K562_100kb = pd.read_csv("/home/coder/data/K562_enhancer_gene_links_100kb.hg38.tsv", sep="\t")
enhancer_gene_GM12878_100kb = pd.read_csv("/home/coder/data/GM12878_enhancer_gene_links_100kb.hg38.tsv", sep="\t")

gene_K562_tss = pd.read_csv("/home/coder/data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt", sep="\t")[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
gene_GM12878_tss = pd.read_csv("/home/coder/data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt", sep="\t")[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]

# CHANGE THIS BACK TO JUST COVERAGE FILES
for file in tqdm(coverage_files, desc="File"):
    cell_line = file[ :file.find(".")]
    file_name = file[ :file.find("E") + 11]
    enhancer_gene_100kb = None
    gene_tss = None

    rna_df = pd.read_csv(f"/home/coder/data/RNASeq_bw/{file}", header=None, sep="\t")

    if cell_line == "K562":
        enhancer_gene_100kb = enhancer_gene_K562_100kb
        gene_tss = gene_K562_tss
    elif cell_line == "GM12878":
        enhancer_gene_100kb = enhancer_gene_GM12878_100kb
        gene_tss = gene_GM12878_tss
    
    gene_tss["ENSID"] = gene_tss["Ensembl_ID"]
    enhancer_gene_100kb_includeNoEnhancerGene = enhancer_gene_100kb.merge(gene_tss, left_on="TargetGeneEnsembl_ID", right_on="Ensembl_ID", how="right", suffixes=["", "_gene"]).reset_index()
    gene_list = list(gene_tss["ENSID"])

    gene_enhancer_table = enhancer_gene_100kb_includeNoEnhancerGene
    rna_method = "encoding"

    rna_df_list = []

    for gene in tqdm(gene_list, desc="Gene"):
        gene_df = gene_enhancer_table[gene_enhancer_table["ENSID"] == gene]
        gene_enhancer_df = gene_df

        if rna_method is not None:
            if rna_method == "encoding" or rna_method == "one-hot":
                gene_rna_df = rna_df[rna_df[3] == gene]
                # rna_df_copy = gene_rna_df.copy()
            if rna_method == "embedding":
                gene_rna_df = rna_df[gene_list.index(gene)]
                # rna_df_copy = gene_rna_df.copy()
            
            # gene_rna_df = encode_promoter_enhancer_links(gene_df, max_seq_len=2000, max_n_enhancer=60, max_distanceToTSS=100_000, add_flanking=False, rna_method=rna_method, rna_df=gene_rna_df)

            # Grabs the DNA sequence.
            fasta_path = "/home/coder/data/hg38.fa"
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
            if rna_method == "embedding":
                gene_rna_df = np.concatenate([gene_rna_df.reshape(1, 125), np.zeros([60, 125])])

            rna_df_list.append(gene_rna_df)

    df = pd.DataFrame(rna_df_list)
    df.to_csv(f"/home/coder/REU/csvs/rna_signal/{file_name}.csv")
    plt.plot(df.columns, df.mean())
    plt.xticks(range(0, 2000, 100))
    plt.xlabel("Position in 2Kbp Promoter Window")
    plt.ylabel("Averaged RNA Signal")
    plt.title(f"Distribution of Averaged RNA Signals for {file_name}")
    plt.savefig(f"/home/coder/REU/plots/{file_name}.average_rna_signal.png")
    plt.clf()