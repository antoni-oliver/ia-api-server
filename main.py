from typing import Union, Literal
from fastapi import FastAPI
from fastapi.encoders import jsonable_encoder
from fastapi.responses import HTMLResponse, StreamingResponse
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
            <ul>
                <li><a href="./sd">Stable Diffusion</a></li>
                <li><a href="./ollama">Ollama</a></li>
            </ul>
        </body>
    </html>
    """

@sd.get("/", response_class=HTMLResponse)
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
            <h1>API Stable Diffusion</h1>
            <h2>Pàgina d'inici</h2>
            <p><a href="./docs">Documentació</a>.</p>
        </body>
    </html>
    """

@ollama.get("/", response_class=HTMLResponse)
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
            <h1>API Stable Diffusion</h1>
            <h2>Pàgina d'inici</h2>
            <p><a href="./docs">Documentació</a>.</p>
        </body>
    </html>
    """

####################
# STABLE DIFFUSION #
####################

class SDTxt2ImgQuery(BaseModel):
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

class SDTxt2ImgResponse(BaseModel):
    images: list[str] = Field(default=[], title="Images", description="The generated images in base64 format")
    parameters: object = Field(default={}, title="Parameters")
    info: str | None = Field(default="", titel="Info")

@sd.post("/txt2img")
async def sd_txt2img(query: SDTxt2ImgQuery) -> SDTxt2ImgResponse:
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8002/sdapi/v1/txt2img", json=json_query)
        json_response = response.json()
        return json_response

class SDInterrogateQuery(BaseModel):
    image: str = Field(default="", title="Image", description="Base-64 encoded image")
    model: str | None = Field(default="clip")

@sd.post("/interrogate")
async def sd_interrogate(query: SDInterrogateQuery) -> str:
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        response = await client.post("http://localhost:8002/sdapi/v1/interrogate", json=json_query)
        json_response = response.json()
        return json_response

@sd.get("/sd-models")
async def sd_models():
    async with httpx.AsyncClient() as client:
        response = await client.get("http://localhost:8002/sdapi/v1/sd-models")
        json_response = response.json()
        for item in json_response:
            if 'filename' in item:
                del item['filename']
        
        return json_response

##########
# OLLAMA #
##########

# Generate a completion.
# Parameters
#    model: (required) the model name
#    prompt: the prompt to generate a response for
#    suffix: the text after the model response
#    images: (optional) a list of base64-encoded images (for multimodal models such as llava)
# Advanced parameters (optional):
#    format: the format to return a response in. Format can be json or a JSON schema
#    options: additional model parameters listed in the documentation for the Modelfile such as temperature
#    system: system message to (overrides what is defined in the Modelfile)
#    template: the prompt template to use (overrides what is defined in the Modelfile)
#    stream: if false the response will be returned as a single response object, rather than a stream of objects
#    raw: if true no formatting will be applied to the prompt. You may choose to use the raw parameter if you are specifying a full templated prompt in your request to the API
#    keep_alive: controls how long the model will stay loaded into memory following the request (default: 5m)
#    context (deprecated): the context parameter returned from a previous request to /generate, this can be used to keep a short conversational memory
# https://github.com/ollama/ollama/blob/main/docs/api.md#generate-a-completion

class OLLAMAGenerateQuery(BaseModel):
    model: str | None = Field(default="", title="Model", max_length=50)
    prompt: str | None = Field(default="", title="Prompt", max_length=65535)
    suffix: str | None = Field(default="", title="Suffix", max_length=65535)
    # images
    format: str | None = Field(default="", title="Format", max_length=10)
    options: object = Field(default={}, title="Options")
    system: str | None = Field(default="", title="System", max_length=65535)
    template: str | None = Field(default="", title="Template", max_length=65535)
    stream: bool | None = Field(default=True, title="Stream")
    raw: bool | None = Field(default=False, title="Raw")
    # keep_alive
    context: list[int] = Field(default=None, title="Context")

def ollama_streaming_call(verb, json, timeout):
    with httpx.stream("POST", "http://localhost:11434/api/" + verb, json=json, timeout=timeout) as response:
        yield from response.iter_raw()

@ollama.post("/generate")
async def ollama_generate(query: OLLAMAGenerateQuery):
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        timeout = httpx.Timeout(120.0, read=None)
        if not query.stream:
            response = await client.post("http://localhost:11434/api/generate", json=json_query, timeout=timeout)
            json_response = response.json()
            return json_response
        else:
            # Streaming
            return StreamingResponse(ollama_streaming_call("generate", json=json_query, timeout=timeout))

class OLLAMAChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"] = Field(default="user", title="Role")
    content: str |None = Field(default="", title="Content", max_length=65535)
    # images
    # tools

class OLLAMAChatQuery(BaseModel):
    model: str | None = Field(default="", title="Model", max_length=50)
    messages: list[OLLAMAChatMessage] | None = Field(default=None, title="Messages")
    # tools: object | None = Field(default={}, title="Tools")
    format: str | None = Field(default="", title="Format", max_length=10)
    options: object = Field(default={}, title="Options")
    stream: bool | None = Field(default=True, title="Stream")
    # keep_alive

@ollama.post("/chat")
async def ollama_chat(query: OLLAMAChatQuery):
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        timeout = httpx.Timeout(120.0, read=None)
        if not query.stream:
            response = await client.post("http://localhost:11434/api/chat", json=json_query, timeout=timeout)
            json_response = response.json()
            return json_response
        else:
            # Streaming
            return StreamingResponse(ollama_streaming_call("chat", json=json_query, timeout=timeout))



class OLLAMAEmbedQuery(BaseModel):
    model: str | None = Field(default="", title="Model", max_length=50)
    input: str | list[str] | None = Field(default="", title="Input")
    # tools: object | None = Field(default={}, title="Tools")
    truncate: bool | None = Field(default=True, title="Truncate")
    options: object = Field(default={}, title="Options")
    # keep_alive

@ollama.post("/embed")
async def ollama_embed(query: OLLAMAEmbedQuery):
    json_query = jsonable_encoder(query)
    async with httpx.AsyncClient() as client:
        timeout = httpx.Timeout(120.0, read=None)
        response = await client.post("http://localhost:11434/api/embed", json=json_query, timeout=timeout)
        json_response = response.json()
        return json_response
