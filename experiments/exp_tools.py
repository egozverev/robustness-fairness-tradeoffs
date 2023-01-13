import numpy as np
import json
import os


class Experiment:
    """
    Class that corresponds to the conducting experiments
    """

    def __init__(self, config_path, output_path, task_id=None):
        """
        :param config_path: path to raw experimental json config
        :param output_path: path where to store json with experimental results
        :param task_id: (optional) task id used in slurm for parallel computation
        """

        with open(config_path, 'r') as openfile:
            self.config = json.load(openfile)
        self.output_path = output_path
        self.task_id = int(task_id) if task_id else None

    def run_experiment(self):
        """
        Performs experiment and saves result to self.output_path
        """
        exp_setup = self._generate_exp_setup(self.config)
        if self.task_id:
            exp_setup["func_args"]["task_id"] = self.task_id
        exp_result = exp_setup["func"](**exp_setup["func_args"])
        for i in range(len(exp_result)):
            exp_result[i] = list(map(float, exp_result[i]))
        exp_log = {
            "name": exp_setup["name"],
            "descr": exp_setup["descr"],
            "result": exp_result
        }
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "w+") as outfile:
            json.dump(exp_log, outfile)


    @staticmethod
    def _get_function_by_path(raw_name):
        """
        Takes a function name and returns functional object
        :param raw_name: function name
        :return: function corresponding to raw_name
        """
        sep = raw_name.rfind(".")
        mod_name = raw_name[:sep]
        func_name = raw_name[sep + 1:]
        mod = __import__(mod_name, fromlist=[""])
        func = getattr(mod, func_name)
        return func

    @staticmethod
    def _generate_exp_setup(config):
        """
        Generates experimental setup from a raw setup json-like config
        :param config: json-like dict with experimental configuration (exp name, descr, exp. function, params)
        :return: setup that contains experimental name, description, functions to be computed and its parameters
        """
        setup = {}
        for key in ["name", 'descr']:
            setup[key] = config[key]
        setup["func"] = Experiment._get_function_by_path(config["func"])
        func_args = {}
        for k, v in config["func_args"].items():
            if k == "model":
                path = "source.models." + v
                func_args[k] = Experiment._get_function_by_path(path)
            elif k == "loss_fncs":
                loss_fncs = [Experiment._get_function_by_path("source.models.torch_compatible_losses." + v[i]) for i in range(len(v))]
                func_args["loss_fncs"] = loss_fncs
            elif k == "metrics_fncs":
                metrics_fncs = [Experiment._get_function_by_path("source.metrics.metrics." + v[i]) for i in range(len(v))]
                func_args['metrics_fncs'] = metrics_fncs
            elif k in ["x", "y", "groups"]:
                func_args[k] = np.load(v)
            else:
                func_args[k] = v
        setup["func_args"] = func_args
        return setup


def aggregate_result(raw_result_path, output_path, task_num):
    """
    Aggregate results from parallel experiments into one file
    :param raw_result_path: path to json results
    :param output_path: path to output file
    :param task_num: number of initial experiments
    """
    aggregated_result = {}

    for i in range(1, task_num + 1):
        with open(raw_result_path + f"{i}.json", 'r') as openfile:
            local_result = json.load(openfile)
        if i == 1:
            aggregated_result["name"] = local_result["name"]
            aggregated_result["descr"] = local_result["descr"]
            aggregated_result["result"] = [[elem] for elem in local_result['result']]
        else:
            for j, elem in enumerate(local_result['result']):
                aggregated_result['result'][j].append(elem)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w+") as outfile:
        json.dump(aggregated_result, outfile)


