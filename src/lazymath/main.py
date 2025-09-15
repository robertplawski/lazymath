# src/lazymath/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pix2text import Pix2Text
import tempfile
import os
import string
import requests
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

app = FastAPI(title="LazyMath API", version="0.1.0")

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("Error, environment variable OPENROUTER_API_KEY not found")
    sys.exit(1)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")


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
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        print(e)
        return None


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    filename = file.filename or "file"
    try:
        suffix = os.path.splitext(filename)[1]
        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp.flush()
            p2t = Pix2Text.from_config()
            outs = p2t.recognize_text_formula(
                tmp.name, resized_shape=768, return_text=False
            )
            # tesseract_text = pytesseract.image_to_string(tmp.name, lang="pol")
            embeddings = [out["text"] for out in outs if out["type"] == "embedding"]

            letters = list(string.ascii_lowercase[:6])  # a to f
            num_letters = len(letters)

            # Assign number + letter
            numbered_lettered = {}
            for i, ex in enumerate(embeddings):
                number = i // num_letters + 1
                letter = letters[i % num_letters]
                key = f"{number}.{letter}"
                numbered_lettered[key] = ex

            exercise_objects = [
                {
                    "id": key,  # e.g., "1.a"
                    "expression": expr,  # the LaTeX expression
                    "solution": None,  # placeholder for solution
                }
                for key, expr in numbered_lettered.items()
            ][:1]

            for obj in exercise_objects:
                obj["solution"] = get_solution(obj["expression"])

        return JSONResponse(
            {
                "filename": os.path.basename(filename),
                "content_type": file.content_type,
                "size": len(content),
                "exercises": exercise_objects,
            }
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Example route
@app.get("/")
async def read_root():
    return {"message": "Pong!"}


# Example route with parameters
@app.get("/add/{a}/{b}")
async def add_numbers(a: int, b: int):
    return {"result": a + b}


# Optional: using a Pydantic model for POST
class Numbers(BaseModel):
    a: int
    b: int


@app.post("/add")
async def add_post(numbers: Numbers):
    return {"result": numbers.a + numbers.b}


def main_function():
    import uvicorn

    uvicorn.run("lazymath.main:app", host="0.0.0.0", port=8000, reload=True)
