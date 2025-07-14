import sys
import os
from tqdm import tqdm
import kipoiseq
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from multiprocessing import Pool
from scripts.utils import FastaStringExtractor, one_hot_encode


def process_gene(gene):
    gene_df = gene_enhancer_table[gene_enhancer_table["ENSID"] == gene]
    gene_enhancer_df = gene_df

    plus_gene = plus[plus[3] == gene]
    minus_gene = minus[minus[3] == gene]

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

    # Grabs the RNA signals from each promoter.
    gene_len = row_0["end"] - row_0["start"]
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

    return plus_gene, minus_gene
    #plus_df_list.append(plus_gene)
    #minus_df_list.append(minus_gene)

    
    
if __name__ == "__main__":    

    coverage_files = glob.glob1("data/RNASeq_bw", "*coverage.txt")
    
    #print(f"Coverage files: {coverage_files}")

    # Generating the RNA signal line plot for each file, split into plus and minus dataframes.
    enhancer_gene_K562_100kb = pd.read_csv("data/K562_enhancer_gene_links_100kb.hg38.tsv", sep="\t")
    enhancer_gene_GM12878_100kb = pd.read_csv("data/GM12878_enhancer_gene_links_100kb.hg38.tsv", sep="\t")
    
    #print("Enhancer Gene Links has been stored")

    gene_K562_tss = pd.read_csv("data/ABC-multiTSS_nominated/K562/Neighborhoods/GeneList.txt", sep="\t")[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]
    gene_GM12878_tss = pd.read_csv("data/ABC-multiTSS_nominated/GM12878/Neighborhoods/GeneList.txt", sep="\t")[['name', 'Ensembl_ID', 'chr', 'tss', 'strand', 'H3K27ac.RPM.TSS1Kb', 'DHS.RPM.TSS1Kb']]

    #print("Gene List has been stored")
    
    fasta_path = "data/hg38.fa"
    fasta_extractor = FastaStringExtractor(fasta_path)

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

        df = pd.read_csv(f"data/RNASeq_bw/{file}", sep="\t", skiprows=3)
        #print("RNASeq File has been stored")
        df.columns = range(len(df.columns))
        plus = df[df[5] == "+"]
        minus = df[df[5] == "-"]

        gene_tss["ENSID"] = gene_tss["Ensembl_ID"]
        enhancer_gene_100kb_includeNoEnhancerGene = enhancer_gene_100kb.merge(gene_tss, left_on="TargetGeneEnsembl_ID", right_on="Ensembl_ID", how="right", suffixes=["", "_gene"]).reset_index()
        gene_list = list(gene_tss["ENSID"])

        gene_enhancer_table = enhancer_gene_100kb_includeNoEnhancerGene
        rna_method = "encoding"

        plus_df_list = []
        minus_df_list = []
        total_df_list = []

        #print("Beginning multiprocessing of genes...")

        pool = Pool(processes=80)
        for gene in tqdm(pool.imap(process_gene, gene_list), total=len(gene_list)):
            plus_gene, minus_gene = gene
            plus_df_list.append(plus_gene)
            minus_df_list.append(minus_gene)

        plus_df = pd.DataFrame(plus_df_list)
        minus_df = pd.DataFrame(minus_df_list)
        total_df = plus_df + minus_df

        plus_df.to_csv(f"data/plots/coverage/{file_name}.plus_rna.csv", index=False, header=False)
        minus_df.to_csv(f"data/plots/coverage/{file_name}.minus_rna.csv", index=False, header=False)
        total_df.to_csv(f"data/plots/coverage/{file_name}.total_rna.csv", index=False, header=False)

        # Plotting just the both dataframes against each other.
        plt.plot(plus_df.columns, plus_df.mean())
        plt.plot(minus_df.columns, minus_df.mean())
        plt.xticks(range(0, 2000, 200))
        plt.xlabel("Position in 2Kbp Promoter Window")
        plt.ylabel("Averaged RNA Signal")
        plt.title(f"Distribution of Averaged RNA Signals for {file_name} Plus and Minus Strands", fontsize=8)
        plt.savefig(f"data/plots/coverage/{file_name}.plus_minus.average_rna_signals.png")
        plt.clf()

     
        # Plotting just the both dataframes against each other.
        plt.plot(total_df.columns, total_df.mean())
        plt.xticks(range(0, 2000, 200))
        plt.xlabel("Position in 2Kbp Promoter Window")
        plt.ylabel("Averaged RNA Signal")
        plt.title(f"Distribution of Averaged RNA Signals for {file_name}", fontsize=8)
        plt.savefig(f"data/plots/coverage/{file_name}.average_rna_signals.png")
        plt.clf()
