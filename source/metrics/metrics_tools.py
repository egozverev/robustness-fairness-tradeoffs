def compute_metrics(metrics, params):
    """
    Computes metrics values for given params
    :param metrics: list of metric functions
    :param params: parameters to be passed in metrics
    :return: list if metric values for given params
    """
    return [m(params) for m in metrics]