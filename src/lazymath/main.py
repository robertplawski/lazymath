# src/lazymath/main.py
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="LazyMath API", version="0.1.0")


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
