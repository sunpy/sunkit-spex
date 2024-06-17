def minimize_func(params, data_y, model_x, model_func, statistic_func):
    """ 
    Minimization function. 
    
    Chosen via likelihood/fit. stat. chosen.
    """
    model_y = model_func.evaluate(model_x, *params)
    return statistic_func(data_y, model_y)