

def scipy_minimize(objective_func, param_guesses, *args, **kwargs):

    method = kwargs.get("method", "Nelder-Mead")

    return minimize(objective_func, 
               param_guesses, 
               args=args,
               method=method)