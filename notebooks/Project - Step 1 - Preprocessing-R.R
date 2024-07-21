library("edgeR")
library("matrixStats")
library(stringr)

setwd("../human_MTG_gene_expression_matrices_2018-06-14")

# Load data
intron_matrix <- read.csv("human_MTG_2018-06-14_intron-matrix.csv")
exon_matrix <- read.csv("human_MTG_2018-06-14_exon-matrix.csv")
full_matrix <- intron_matrix + exon_matrix

# Normalize data
full_matrix_log2cpm <- log2(cpm(full_matrix) + 1)

# Replace first row of gene IDs
full_matrix[,1] <- intron_matrix[,1]
full_matrix_log2cpm[,1] <- intron_matrix[,1]

# Load sample labels
samples <- read.csv("human_MTG_2018-06-14_samples-columns.csv", stringsAsFactors = FALSE)
labels <- samples[c("sample_name", "cluster")]
labels <- labels[labels$cluster != "no class",]
labels <- cbind(labels, str_split_fixed(labels$cluster, " ", n=Inf))
colnames(labels) <- c("sample_name", "cluster", "higher", "layer", "intermediate", "granular")

cell.types <- unique(labels$cluster)

# Identify columns for each cell type
for(type in cell.types) {
  t(assign(paste0(type, ".celltypes"), labels[which(labels$cluster == type),1]))
}

# Load gene names
gene.rows <- read.csv("human_MTG_2018-06-14_genes-rows.csv")
rownames(full_matrix_log2cpm) <- gene.rows[["gene"]]

# Remove genes with 0 expression
keep <- rowSums(full_matrix_log2cpm[, 2:15929]) > 0
kept.genes <- full_matrix_log2cpm[keep,]

# Subset data for each cell type
for (n in cell.types) {
  temp.cell.types <- get(paste0(n, ".celltypes"))
  temp.subset <- kept.genes[, temp.cell.types]
  assign(paste0(n, ".mm.subset"), temp.subset)
}

# Create median common gene matrix
cg.median.mm.df <- data.frame(matrix(ncol = 0, nrow = length(kept.genes[,1])))
rownames(cg.median.mm.df) <- rownames(kept.genes)

# Fill median common gene matrix
for (n in cell.types) {
  temp.subset.common <- get(paste0(n, ".mm.subset"))
  temp.subset.median.common <- rowMedians(temp.subset.common)
  cg.median.mm.df <- cbind(cg.median.mm.df, temp.subset.median.common)
}

colnames(cg.median.mm.df) <- cell.types

# Calculate variance and remove genes with 0 variance
cg.median.mm.var <- apply(cg.median.mm.df, 1, var)
keep.var <- which(cg.median.mm.var != 0)
cg.median.mm.var <- cg.median.mm.var[cg.median.mm.var != 0]

# Subset genes after removing variance criteria
cg.median.mm.df.var <- cg.median.mm.df[keep.var,]

# Create final matrix for machine learning
full_matrix_log2cpm_subset <- rbind(subset(full_matrix_log2cpm, select = -X), samples[["cluster"]])
keep.var.genes <- names(keep.var)
rownames(full_matrix_log2cpm_subset)[50282] <- "Classification"
final.matrix <- full_matrix_log2cpm_subset[c(keep.var.genes, "Classification"),]

# Save final matrix to CSV
write.csv(final.matrix, file = "everything.csv", row.names = TRUE, col.names = TRUE)
