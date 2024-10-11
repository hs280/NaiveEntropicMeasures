import ray
import time

# Initialize Ray with the dashboard
ray.init(dashboard_port=8265)

@ray.remote
def f(x):
    time.sleep(1)
    return x

# Launch several tasks
futures = [f.remote(i) for i in range(int(1000))]

# Get the results
results = ray.get(futures)

print(results)
