# Data Origins

### Competition:
From the original competition page: https://www.kaggle.com/competitions/stanford-rna-3d-folding/data

### Structures:
From paper: https://pubmed.ncbi.nlm.nih.gov/37838833/, they provided the code here: https://bitbucket.org/dokhlab/eprna-euclidean-parametrization-of-rna/src/master/structures/
which is extracted from the rcsb pdb: https://www.rcsb.org/

### Scraping:
Scraped the ICSB protein databank for RNA-only structures with a resolution of less than 5.0 angstroms. (Inspired by https://pubmed.ncbi.nlm.nih.gov/37838833/)

# Combined:
Contains all combined samples from 'Competition', 'Structures' and 'Scraping', all duplicates removed and stored as Pytorch Tensor.
-  "rna_id": The pbd-id of the RNA molecule
- "sequence": The encoded sequence of the RNA molecule ({'G': 1, 'U': 2, 'A': 3, 'C': 4})
- "label": The 3d structure tensor of the RNA molecule after folding.