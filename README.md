# ScCCL

ScCCL:Single-cell data clustering based on self-supervised contrastive learning

ScCCL, a novel self-supervised contrastive learning method for clustering of scRNA-seq data. For scRNA-seq data, ScCCL first randomly masks the gene expression of each cell twice and adds a small amount of Gaussian noise, and then uses the momentum encoder structure to extract features from the enhanced data. Contrastive learning is then applied in the instance-level contrastive learning module and the cluster-level contrastive learning module, respectively. 

Due to the size limitation, only one dataset is uploaded on the warehouse, and other data are exposed in the Google cloud disk: https://drive.google.com/drive/folders/1tl2VgoEQdgR4p2PiCgYGHAGzgGqskR7P?usp=sharing
