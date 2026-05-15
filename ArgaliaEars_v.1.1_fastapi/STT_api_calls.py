from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
import numpy as np
from vadVer import Transcriber
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

runData = True

@asynccontextmanager
async def lifespan(app: FastAPI):
    global runData
    print("Configuating.......")
    # app.state.calls = STT_handler("base")
    app.state.calls = Transcriber()
    yield
    runData = False
    print("Shutting down")

# @app.get("/")
# async def get():
#     return HTMLResponse(html)
# 16 bits pcm encoding (2 bytes), means it should send 8192 bytes->4096 samples. 4096/16000 -> 0.25 secs each frame

app = FastAPI(lifespan=lifespan)

app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],           # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# next ver, add a buffer here, so send later instead of constantly call main
@app.websocket("/ws")
async def websocket_end(websocket: WebSocket):
    global runData
    await websocket.accept()
    try:
        while runData:
            # Receive raw PCM bytes from frontend
            data = await websocket.receive_bytes()
            theNum = np.frombuffer(data, dtype=np.int16).astype(np.float32)
            # print(f"{theNum}")
            # await websocket.send_text(f"the data is: {len(theNum)} long")
            datas = await app.state.calls.main(theNum, websocket)

            # Here you can handle the PCM data:
            # - Append to a buffer
            # - Save to a WAV file
            # - Stream into a speech recognition model
            # For now, just echo back the size of the chunk
            # await websocket.send(datas)
            # print(datas)
            # if(result!=None):
            #     print("yay")
    except Exception as e:
        print(f"WebSocket closed: {e}")
