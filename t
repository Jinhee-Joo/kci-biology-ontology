from google import genai
client = genai.Client()
try:
    r = client.models.generate_content(model="models/gemini-2.0-flash", contents="ping")
    print("OK:", r.text[:50])
except Exception as e:
    print("FAIL:", e)
