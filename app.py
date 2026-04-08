
import requests, base64

invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
stream = False


headers = {
  "Authorization": "Bearer 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",
  "Accept": "text/event-stream" if stream else "application/json"
}

payload = {
  "model": "meta/llama-4-maverick-17b-128e-instruct",
  "messages": [{"role":"user","content":"Provide an aricle on Machine Learning"}],
  "max_tokens": 512,
  "temperature": 1.00,
  "top_p": 1.00,
  "frequency_penalty": 0.00,
  "presence_penalty": 0.00,
  "stream": stream
}

response = requests.post(invoke_url, headers=headers, json=payload)

if stream:
    for line in response.iter_lines():
        if line:
            print(line.decode("utf-8"))
else:
    print(response.json())
