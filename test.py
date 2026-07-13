import os
import time
from openai import OpenAI
from dotenv import load_dotenv
 
load_dotenv()
 
API_KEY = os.getenv("API_KEY","sk-PK8q5nvwPI2d2Zz-jNK9iw")
BASE_URL = os.getenv("GENAI_SHARED_SERVICE_BASE","https://genai-sharedservice-americas.pwc.com")
MODEL = os.getenv("GENAI_SHARED_SERVICE_MODEL", "azure.gpt-4o")
 
client = OpenAI(
    api_key=API_KEY,
    base_url=f"{BASE_URL}/openai/v1"
)
 
start_time = time.perf_counter()
 
response = client.chat.completions.create(
    model=MODEL.replace("azure.", ""),
    messages=[
        {
            "role": "system",
            "content": "You are a helpful assistant."
        },
        {
            "role": "user",
            "content": "Explain DORA regulation in 5 bullet points."
        }
    ],
    max_tokens=500,
    temperature=0.2
)
 
end_time = time.perf_counter()
 
total_time = end_time - start_time
 
print("========== API Timing Result ==========")
print(f"Model Used     : {MODEL}")
print(f"Time Taken     : {total_time:.2f} seconds")
print("=======================================")
 
print("\nResponse:")
print(response.choices[0].message.content)