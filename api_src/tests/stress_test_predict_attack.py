import httpx
import os
import asyncio
import time
import matplotlib.pyplot as plt
from io import BytesIO

# Path to your existing Parquet file
path = os.path.join("api_src", "tests", "10_samples.parquet")

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8002/predict-attack"

async def send_request(client):
    # Create a new buffer for each request
    with open(path, "rb") as file:
        parquet_buffer = BytesIO(file.read())

    # Send a POST request with the Parquet file and measure the response time
    files = {'file': ("10_samples.parquet", parquet_buffer, "application/octet-stream")}
    start_time = time.time()
    response = await client.post(url, files=files)
    end_time = time.time()
    duration = end_time - start_time
    return response.status_code, duration

async def stress_test(num_clients, num_repeats):
    durations = []
    async with httpx.AsyncClient() as client:
        for _ in range(num_repeats):
            # Run multiple clients concurrently
            tasks = [send_request(client) for _ in range(num_clients)]
            results = await asyncio.gather(*tasks)

            # Collect response times and statuses
            for status, duration in results:
                if status == 200:
                    durations.append(duration)

    return durations

def plot_performance(durations):
    # Plot the response durations
    plt.figure(figsize=(10, 6))
    plt.plot(durations, marker="o", linestyle="--")
    plt.xlabel("Request Number")
    plt.ylabel("Response Time (seconds)")
    plt.title("Server Response Time for Concurrent Requests")
    plt.show()

# Parameters for the stress test
num_clients = 100  # Number of simultaneous clients
num_repeats = 4    # Number of times the clients should repeat the test

# Run the stress test and plot the results
durations = asyncio.run(stress_test(num_clients, num_repeats))
plot_performance(durations)
