# src/lazymath/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from pix2text import Pix2Text
import tempfile
import os
import string
import requests
import json
import sys
import asyncio
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI(title="LazyMath API", version="0.1.0")

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("Error, environment variable OPENROUTER_API_KEY not found")
    sys.exit(1)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

@app.get("/solution")
def get_solution(expression: str):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer %s" % OPENROUTER_API_KEY,
        },
        data=json.dumps(
            {
                "model": "deepseek/deepseek-chat-v3.1:free",
                "messages": [
                    {
                        "role": "system",
                        "content": "Output only solution steps without any text or explanations. Write only in latex.",
                    },
                    {"role": "user", "content": expression},
                ],
            }
        ),
    )
    result = response.json()
    try:
        content = result["choices"][0]["message"]["content"]
        return content
    except Exception as e:
        print(e)
        return None

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    filename = file.filename or "file"
    try:
        suffix = os.path.splitext(filename)[1]
        content = await file.read()

        async def event_stream():
            with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
                tmp.write(content)
                tmp.flush()

                p2t = Pix2Text.from_config()
                outs = p2t.recognize_text_formula(tmp.name, resized_shape=768, return_text=False)
                embeddings = [out["text"] for out in outs if out["type"] == "embedding"]

                letters = list(string.ascii_lowercase[:6])
                num_letters = len(letters)

                # Assign number + letter
                numbered_lettered = {}
                for i, ex in enumerate(embeddings):
                    number = i // num_letters + 1
                    letter = letters[i % num_letters]
                    key = f"{number}.{letter}"
                    numbered_lettered[key] = ex

                # Stream metadata first
                yield json.dumps({
                    "filename": os.path.basename(filename),
                    "content_type": file.content_type,
                    "size": len(content),
                    "exercises": []
                }) + "\n"

                # Sequentially process each exercise
                for key, expr in numbered_lettered.items():
                    solution = get_solution(expr)  # blocking call
                    exercise_obj = {
                        "id": key,
                        "expression": expr,
                        "solution": solution
                    }
                    yield json.dumps(exercise_obj) + "\n"

        return StreamingResponse(event_stream(), media_type="application/json")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




def main_function():
    import uvicorn

    uvicorn.run("lazymath.main:app", host="0.0.0.0", port=8000, reload=True)
