import httpx
import os
from io import BytesIO

# Path to your existing Parquet file
path = os.path.join("api_src", "tests" , "10_samples.parquet")

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8002/predict-attack"

# Read the Parquet file into a buffer
with open(path, "rb") as file:
    parquet_buffer = BytesIO(file.read())

# Send a POST request with the Parquet file
async def test_predict_attack():
    async with httpx.AsyncClient() as client:
        files = {'file': ("10_samples.parquet", parquet_buffer, "application/octet-stream")}
        response = await client.post(url, files=files)

        # Print response status and JSON output
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

# Run the test
import asyncio
asyncio.run(test_predict_attack())
