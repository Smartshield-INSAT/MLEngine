import httpx
import os
import asyncio
import time
import matplotlib.pyplot as plt
from io import BytesIO

# Path to your existing Parquet file
path = os.path.join("api_src", "tests", "10_samples.parquet")

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8002/predict-all"

async def send_request(client):
    # Create a new buffer for each request to ensure safe concurrent execution
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
    success_count = 0
    error_count = 0

    async with httpx.AsyncClient() as client:
        for _ in range(num_repeats):
            # Run multiple clients concurrently
            tasks = [send_request(client) for _ in range(num_clients)]
            results = await asyncio.gather(*tasks)

            # Collect response times and statuses
            for status, duration in results:
                durations.append(duration)
                if status == 200:
                    success_count += 1
                else:
                    error_count += 1

    return durations, success_count, error_count

def plot_performance(durations, success_count, error_count):
    # Plot the response durations and success/error rates
    plt.figure(figsize=(12, 10))

    # Plot response times
    plt.subplot(2, 1, 1)
    plt.plot(durations, marker="o", linestyle="--", color="blue")
    plt.xlabel("Request Number")
    plt.ylabel("Response Time (seconds)")
    plt.title("Server Response Time for Concurrent Requests")

    # Plot success and error rate
    plt.subplot(2, 1, 2)
    total_requests = success_count + error_count
    success_rate = (success_count / total_requests) * 100 if total_requests else 0
    error_rate = (error_count / total_requests) * 100 if total_requests else 0
    plt.bar(["Success Rate", "Error Rate"], [success_rate, error_rate], color=["green", "red"])
    plt.ylabel("Percentage (%)")
    plt.title("Success and Error Rate")

    plt.tight_layout()
    plt.show()

# Parameters for the stress test
num_clients = 100  # Number of simultaneous clients
num_repeats = 4    # Number of times the clients should repeat the test

# Run the stress test and plot the results
durations, success_count, error_count = asyncio.run(stress_test(num_clients, num_repeats))
plot_performance(durations, success_count, error_count)
