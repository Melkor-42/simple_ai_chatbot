from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
import uvicorn
from utils import get_response, predict_class

app = FastAPI()
templates = Jinja2Templates(directory="templates")


class MessageRequest(BaseModel):
    message: str


@app.get("/")
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/handle_message")
async def handle_message(request_data: MessageRequest):
    message = request_data.message
    intents_list = predict_class(message)
    response = get_response(intents_list)
    return JSONResponse(content={"response": response})


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
