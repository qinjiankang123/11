import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from openai import OpenAI
import numpy as np

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# -------------------------- 星河科技知识库 --------------------------
KNOWLEDGE = """【公司介绍】
星河科技有限公司是一家面向中小企业提供智能客服、知识库问答和数据分析工具的软件公司。

【产品套餐】
星河智能客服目前有三个套餐。基础版价格为每月 99 元，适合个人和小团队使用。专业版价格为每月 299 元，适合成长型企业使用。企业版需要联系销售定制报价。

【发票政策】
用户付款成功后，可以在后台的“费用中心”申请电子发票。普通发票通常会在 1 个工作日内开具，专用发票通常会在 3 个工作日内开具。

【退款规则】
月付套餐购买后 7 天内，如果 AI 问答使用量少于 50 次，可以申请退款。超过 7 天或使用量达到 50 次及以上，不支持退款。

【人工客服】
如果智能客服无法解决问题，用户可以通过官网右下角的在线客服入口联系人工客服。人工客服工作时间为周一至周五 9:00-18:00。"""

def load_knowledge():
    chunks = []
    for block in KNOWLEDGE.split("\n\n"):
        chunk = block.strip()
        if chunk:
            chunks.append(chunk)
    return chunks

def get_embeddings(texts):
    response = client.embeddings.create(
        model="text-embedding-v4",
        input=texts
    )
    return [item.embedding for item in response.data]

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve(query, chunks, vectors):
    q_vec = get_embeddings([query])[0]
    scored = []
    for c, v in zip(chunks, vectors):
        scored.append((cosine_similarity(q_vec, v), c))
    scored.sort(reverse=True)
    return scored[:3]

chunks = load_knowledge()
vectors = get_embeddings(chunks)

# -------------------------- 聊天接口 --------------------------
class ChatMessage(BaseModel):
    message: str

@app.post("/chat")
async def chat(msg: ChatMessage):
    # 先做RAG检索
    results = retrieve(msg.message, chunks, vectors)
    context = "\n".join([c for _, c in results])

    prompt = f"""你是星河科技的企业智能客服。
请只根据下面的资料回答用户问题，不要编造。
如果资料里没有答案，就回复：“知识库中暂未找到相关信息，需要人工客服进一步处理”。

资料：
{context}

用户问题：{msg.message}"""

    completion = client.chat.completions.create(
        model="qwen-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    reply = completion.choices[0].message.content
    return {"reply": reply}

@app.get("/")
async def root():
    return {"message": "请访问 /static/index.html"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)