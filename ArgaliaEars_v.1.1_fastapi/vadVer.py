import numpy as np
import asyncio
import math
from collections import deque
from silero_vad import load_silero_vad, VADIterator
import asyncio
from faster_whisper import WhisperModel
from fastapi import WebSocket

WHISPERVERSION = "large-v3-turbo"
SAMPLERATE = 16000
CONVERSLEN = 5 


class Transcriber:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.arrayOne = []
        # this will become a buffer reserver later
        theVad = load_silero_vad()
        self.model = VADIterator(theVad, threshold=0.5, sampling_rate=16000, min_silence_duration_ms=500)
        
        self.rolling_buffer = deque(maxlen=CONVERSLEN)
        print("Model found, using "+WHISPERVERSION)
        # cuda if nvidia
        self.modules = WhisperModel(WHISPERVERSION, device="cpu", compute_type="float32")
        self.vadCut = deque()
        self.speechActi = False
    
    async def main(self, data: np.frombuffer, ws:WebSocket):
        deriazation = self.model(
            data
         )
        # print(len(self.vadCut))
        if(deriazation):
            if("start" in deriazation):
                self.speechActi = True
                print("working")
            elif("end" in deriazation):
                self.speechActi = False
                print("not working")
                self.arrayOne = np.array([], np.float32)
                self.arrayOne = np.concatenate(self.vadCut).astype(np.float32)
                self.vadCut.clear()
                if not self.lock.locked():
                    asyncio.create_task(
                        self.transcribing(ws)
                    )

        if(self.speechActi):
            self.vadCut.append(data)

    async def transcribing(self, ws:WebSocket):
        async with self.lock:
            segment, result = self.modules.transcribe(
                self.arrayOne, 
                beam_size=5, 
                vad_filter=False, 
                condition_on_previous_text=False,   # very important for streaming
                word_timestamps=False,      # set True only if you need them (slows it down)
                temperature=0.0, language="en")

            retData = ""
            for seg in segment:
                retData+=seg.text

            print(retData)
            await ws.send_json(
                {"text": retData}
            )


