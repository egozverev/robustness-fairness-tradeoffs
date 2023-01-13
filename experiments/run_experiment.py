from experiments.exp_tools import Experiment
import sys

if __name__ == '__main__':
    exp = Experiment(*sys.argv[1:])
    exp.run_experiment()
