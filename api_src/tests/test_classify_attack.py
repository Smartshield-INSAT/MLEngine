import httpx
import os
from io import BytesIO

# Path to your existing Parquet file
path = os.path.join("api_src", "tests" , "10_attack_cat_samples.parquet")

# Open the Parquet file
with open(path, "rb") as file:
    parquet_file = file.read()

# Define the URL of the FastAPI endpoint
url = "http://127.0.0.1:8002/predict-attack-cat"

# Send a POST request with the Parquet file
async def test_predict_attack_cat():
    async with httpx.AsyncClient() as client:
        files = {'file': ("10_attack_cat_simples.parquet", parquet_file, "application/octet-stream")}
        response = await client.post(url, files=files)

        # Print response status and JSON output
        print("Status Code:", response.status_code)
        print("Response JSON:", response.json())

# Run the test
import asyncio
asyncio.run(test_predict_attack_cat())