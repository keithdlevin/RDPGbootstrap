# RDPGbootstrap
Method as described in <a href='https://arxiv.org/abs/1907.10821'>Bootstrapping Networks with Latent Space Structure</a> by Levin and Levina.

Different samplers as described in the paper are implemented in the file `netboot.py`. These samplers are initialized by passing in the observed network (encoded in its adjacency matrix `A`) and specifying an embedding dimension.
For example, `netboot.ASENetSampler(A, 3)` will initilize a sampler with latent positions estimated via ASE, embedding into 3 dimensions.
