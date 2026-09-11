"""
FastAPI app entry point. Run with: uvicorn api.main:app --reload
Then point smee.io (or ngrok) at this server's /webhooks/github route.
"""
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI

from api.webhooks import router

app = FastAPI()
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok"}
