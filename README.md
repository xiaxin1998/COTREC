# COTREC
Codes for CIKM'21 paper 'Self-Supervised Graph Co-Training for Session-based Recommendation'.

Requirements: Python 3.7, Pytorch 1.6.0, Numpy 1.18.1

Best Hyperparameter:
+ Tmall: beta=0.01, alpha=0.005, eps=0.2
+ RetailRocket: beta=0.01, alpha=0.005, eps=0.2
+ Diginetica: beta=0.001, alpha=0.005, eps=0.5

Datasets are available at Dropbox: https://www.dropbox.com/sh/j12um64gsig5wqk/AAD4Vov6hUGwbLoVxh3wASg_a?dl=0 The datasets are already preprocessed and encoded by pickle.

Some people may encounter a cudaError in line 50 or line 74 when running our codes if your numpy and pytorch version are different with ours. Currently, we haven't found the solution to resolve the version problem. If you have this problem, please try to change numpy and pytorch version same with ours.
