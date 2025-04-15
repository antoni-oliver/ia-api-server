from typing import Union
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import httpx
import mariadb
import sys
import os

try:
    conn = mariadb.connect(
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        host=os.environ['DB_HOST'],
        port=int(os.environ['DB_PORT']),
        database=os.environ['DB_DATABASE']
    )
except mariadb.Error as e:
    print(f"Error connecting to MariaDB Platform: {e}")
    sys.exit(1)

dbCursor = conn.cursor()

print(dbCursor)

app = FastAPI()

api = FastAPI()
app.mount('/api', api)

sd = FastAPI()
api.mount('/sd', sd)

ollama = FastAPI()
api.mount('/ollama', ollama)


@app.get("/", response_class=HTMLResponse)
def app_root():
    #return "Servidor IA-LTIM. Prova d'accedir a /api."
    return """
    <!doctype html>
    <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <title>IA-LTIM</title>
        </head>
        <body>
            <h1>Servidor IA-LTIM</h1>
            <h2>Pàgina d'inici</h2>
            <p>L'API està a <a href="./api">/api</a>.</p>
        </body>
    </html>
    """

@api.get("/", response_class=HTMLResponse)
def api_root():
    return """
    <!doctype html>
    <html>
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width,initial-scale=1">
            <title>IA-LTIM</title>
        </head>
        <body>
            <h1>API IA-LTIM</h1>
            <h2>Pàgina d'inici</h2>
            <p>Consulta la documentació a <a href="./docs">/docs</a>.</p>
        </body>
    </html>
    """
####
class Txt2ImgQuery(BaseModel):
    prompt: str | None = Field(default="", title="Prompt", max_length=250)
    negative_prompt: str | None = Field(default="", title="Negative prompt", max_length=250)
    model: str | None = Field(default="", title="Model", max_length=50)
    seed: int | None = Field(default=-1, title="Seed")
    sampler_name: str | None = Field(default="", title="Sampler name", max_length=50)
    batch_size: int | None = Field(default=1, title="Batch size")
    n_iter: int | None = Field(default=1, title="N iter")
    steps: int | None = Field(default=50, title="Steps")
    cfg_scale: int | None = Field(default=7, title="CFG scale")
    width: int | None = Field(default=512, title="Width")
    height: int | None = Field(default=512, title="Height")
    sampler_index: str | None = Field(default="Euler", title="Sampler index")

class Txt2ImgResponse(BaseModel):
    images: list[str] = Field(default=[], title="Images", description="The generated images in base64 format")
    parameters: object = Field(default={}, title="Parameters")
    info: str | None = Field(default="", titel="Info")

@api.post("/txt2img")
async def txt2img(query: Txt2ImgQuery) -> Txt2ImgResponse:
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:7860/sdapi/v1/txt2img", json=json_query)
        json_response = response.json()
        return json_response

class InterrogateQuery(BaseModel):
    image: str = Field(default="", title="Image", description="Base-64 encoded image")
    model: str | None = Field(default="clip")

@api.post("/interrogate")
async def interrogate(query: InterrogateQuery) -> str:
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:7860/sdapi/v1/interrogate", json=json_query)
        json_response = response.json()
        return json_response

@api.get("/sd-models")
async def sd_models():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:7860/sdapi/v1/sd-models")
        json_response = response.json()
        for item in json_response:
            if 'filename' in item:
                del item['filename']
        
        return json_response
