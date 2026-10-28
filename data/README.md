# Colon gene-expression data

The real-data experiment uses the Colon gene-expression matrix introduced by Alon et al. (1999), containing 62 samples and 2000 genes. The data file is **not redistributed in this repository** because the source mirror does not state an explicit data license.

## Download instructions

1. Open the source used by the manuscript:

   <https://github.com/alxiang/Colon-Cancer-Classification/blob/master/ColonX.csv>

2. Select **Download raw file**.

3. Save the downloaded file with the exact path and name:

   ```text
   data/ColonX.csv
   ```

4. From the repository root, run the real-data experiment:

   ```matlab
   run(fullfile('demos','demo_Table5_Colon.m'))
   ```

If the file is absent, `Gen_SPCA_Realworld.m` stops with a message containing the download URL and required destination.

## Attribution

U. Alon, N. Barkai, D. A. Notterman, K. Gish, S. Ybarra, D. Mack, and A. J. Levine, “Broad patterns of gene expression revealed by clustering analysis of tumor and normal colon tissues probed by oligonucleotide arrays,” *Proceedings of the National Academy of Sciences*, 96(12), 6745--6750, 1999.

The generator removes sample identifiers and empty variables, replaces non-finite entries by their column mean, and discards constant variables. It then centers each gene across samples, scales it to unit sample variance, and forms L = 2*A'*A/(m-1).
