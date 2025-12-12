
import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv(".env.local")

key = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=key)

print("Available Models:")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

print("\nTrying gemini-pro...")
try:
    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content("Hi")
    print("Success with gemini-pro")
except Exception as e:
    print(f"gemini-pro failed: {e}")
