#%%
from config import HE, SNR, model, data, A, sv 
from timeit import Timer

num_runs = 10000

y,x = next(iter(data))

def evaluate_model_timing():
    x = model(y)
    pass

if __name__ == "__main__":

    duration = Timer(evaluate_model_timing).timeit(number = num_runs)
    avg_duration = duration / num_runs

    print(f"Average duration for model evaluation: " + "{:.3e}".format(avg_duration) + f" seconds")
# %%
