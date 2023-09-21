# Databricks notebook source
import timeit
import mlflow
import pandas as pd

model_uri = "models:/sentiment_bert/latest"
loaded_model=mlflow.pyfunc.load_model(model_uri)

request = pd.DataFrame(["hey, how are you"])

def predict():
    return loaded_model.predict(request)

# COMMAND ----------

# Number of loops (e.g., 1000 times)
num_loops = 1000

# Time the prediction
time_taken = timeit.timeit(predict, number=num_loops)

print(f"Time taken for {num_loops} predict calls: {time_taken:.4f} seconds")
print(f"Average time per predict call: {time_taken/num_loops:.6f} seconds")

# COMMAND ----------

import cProfile
import pstats

profile = cProfile.Profile()
profile.enable()

# Run the predict method 1000 times
loaded_model.predict(request)

profile.disable()

print("=" * 20 + " Read this doc for how to interpret the result: https://docs.python.org/3/library/profile.html")

stats = pstats.Stats(profile).sort_stats('time', 'cumulative')
stats.print_stats(100)
