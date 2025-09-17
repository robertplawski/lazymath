# src/lazymath/main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
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
from sympy import symbols, Eq, sympify, solve

app = FastAPI(title="LazyMath API", version="0.1.0")

load_dotenv()

if not os.getenv("OPENROUTER_API_KEY"):
    print("Error, environment variable OPENROUTER_API_KEY not found")
    sys.exit(1)

OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY")

def llm_complete(system_prompt, user_content, model):
    response = requests.post(
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": "Bearer %s" % OPENROUTER_API_KEY,
        },
        data=json.dumps(
            {
                "seed":0,
                "sort":"throughput",
                "temperature":0.5,
                "model": model, #"deepseek/deepseek-r1-0528-qwen3-8b:free",#"tngtech/deepseek-r1t-chimera:free", #"microsoft/mai-ds-r1:free", #"deepseek/deepseek-r1-0528-qwen3-8b:free",#"deepseek/deepseek-chat-v3.1:free",
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {"role": "user", "content": user_content},
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


# {"task","solutions","domain","steps":[{"name","content"}]}

def parse_equation(eq_str: str, symbol_str: str = "x"):
    x = symbols(symbol_str)
    
    # Split at '='
    if "=" not in eq_str:
        raise ValueError("Equation must contain '='")
    
    lhs_str, rhs_str = eq_str.split("=")
    
    # Convert sides to SymPy expressions
    lhs = sympify(lhs_str)
    rhs = sympify(rhs_str)
    
    # Create SymPy equation
    equation = Eq(lhs, rhs)
    return equation, x
def solve_equation_str(eq_str: str):
    equation, x = parse_equation(eq_str)
    
    # Solve equation
    solution = solve(equation, x)
    
    # Convert solution to Python native types
    solution_safe = [float(s) if s.is_Float else int(s) for s in solution]
    return solution_safe

@app.get("/fast")
def fast(expression: str):
    return solve_equation_str(expression)

@app.get("/json_solution")
def get_json_solution(expression: str):
    return get_json(get_solution(expression), '{"task","solutions","roots","excluded_roots","domain","steps":[{"name","content"}]}')

@app.get("/json")
def get_json(content: str, structure: str):
    return llm_complete("Return json string based on structure: %s" % structure, content, 'openai/gpt-oss-20b:free')


@app.get("/solution")
def get_solution(expression: str):
    return llm_complete("Output only solution steps without any text or explanations. Remember to write the domain D near the end and solve then check whether the solution belongs in domain. Do not write any instructions, write only in latex without any strings use only math symbols.", expression, 'deepseek/deepseek-chat-v3.1:free')
   


class Task(BaseModel):
    section: int
    task: str
    expression: str
    solution: str | None

@app.post("/get_tasks_from_image")
async def get_tasks_from_image(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type")

    filename = file.filename or "file"
    try:
        suffix = os.path.splitext(filename)[1]
        content = await file.read()

        with tempfile.NamedTemporaryFile(delete=True, suffix=suffix) as tmp:
            tmp.write(content)
            tmp.flush()

            p2t = Pix2Text.from_config()
            outs = p2t.recognize_text_formula(tmp.name, resized_shape=768, return_text=False)
            embeddings = [out["text"] for out in outs if out["type"] == "embedding"]

            letters = list(string.ascii_lowercase[:6])
            num_letters = len(letters)

            # Assign number + letter
            numbered_lettered = []
            for i, ex in enumerate(embeddings):
                
                number = i // num_letters + 1
                letter = letters[i % num_letters]
                numbered_lettered.append(Task(section=number, task=letter, expression=ex, solution=None).dict())

        return JSONResponse(numbered_lettered)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload_file_stream")
async def upload_file_stream(file: UploadFile = File(...)):
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
