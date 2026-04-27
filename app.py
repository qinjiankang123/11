import os
from fastapi import FastAPI, staticfiles
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

app.mount("/static", staticfiles.StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(msg: ChatMessage):
    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": msg.message}]
    )
    reply = completion.choices[0].message.content
    return {"reply": reply}

@app.get("/")
async def root():
    return {"message": "https://你的域名.static/index.html"}