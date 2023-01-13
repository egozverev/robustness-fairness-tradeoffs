
# Exploring Accuracy vs Robustness vs Fairness Trade-offs with Multi-Objective Machine Learning

> Note: this repo contains code and documentation associated with my 2-months long research project at ISTA. 

**Tl;DR**: My goal was to see how multiple ML safety properties interact with each other when enforced simultaneously. I used the framework of multi-objectives machine learning to find that out for robustness and fairness. 

Check [full report](https://github.com/egozverev/robustness-fairness-tradeoffs/blob/main/docs/egor_zverev_robustness_fairness_tradeoffs.pdf). 

### How to reproduce results from the report
0. Install slurm workload manager
1. Create an experimental config as described in `experiments/configs`
2. If executed sequentially, run the following command: `sbatch --export=exp_name='EXP_NAME' sequential_script.job`
3. If executed on server with multiple workers, run the following command: `sbatch --export=exp_name='EXP_NAME' --array=1-N:1 parallel_script.job`, where `N` is the size of experimental grid (`grid_sz` parameter from the config). 
