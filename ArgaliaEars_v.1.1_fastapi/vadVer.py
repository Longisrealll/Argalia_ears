import numpy as np
import sounddevice as sd
import asyncio
import math
from collections import deque
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
import asyncio
from faster_whisper import WhisperModel
from fastapi import WebSocket
import sys

WHISPERVERSION = "large-v3-turbo"
# QUEUELEN is browser dependent, different browser will send the data back differently every second, it is not 
# as stable as audiodevice
QUEUELEN = 10
SAMPLERATE = 16000
CONVERSLEN = 5 

class Transcriber:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.arrayOne = np.ndarray([], dtype=np.float32)
        # this will become a buffer reserver later
        self.reservedArray = np.ndarray([], dtype=np.float32)
        self.model = load_silero_vad()
        
        self.rolling_buffer = deque(maxlen=QUEUELEN)
        print("Model found, using "+WHISPERVERSION)
        # cuda if nvidia
        self.modules = WhisperModel(WHISPERVERSION, device="cpu", compute_type="float32")
        self.counters = False
        self.vadCut = deque(maxlen=CONVERSLEN)
        self.safeLock = True

    
    async def main(self, data: np.frombuffer, ws:WebSocket):
        # print("good")

        # while True:
        # print(len(data))

        self.rolling_buffer.append(data)
        # print(data)
            # instead of this, make it run every 3 seconds
        if(self.safeLock):
            # print("working")
            self.safeLock = False
            asyncio.create_task(self.counter())
        # if(False):
        if(self.counters):
            self.counters = False

            for i in range(len(self.rolling_buffer)):
                self.arrayOne = np.concatenate([self.arrayOne, list(self.rolling_buffer[i])], axis=None)
            self.checkingData(ws)
    
    async def counter(self):
        for i in range(1):
            await asyncio.sleep(1)

        self.safeLock = True
        self.counters = True

    def checkingData(self, ws: WebSocket):
        deriazation = get_speech_timestamps(
            self.arrayOne,
            self.model,
            sampling_rate=SAMPLERATE,
            min_silence_duration_ms=700,
            return_seconds=True
        )
        # print(deriazation)
        if(not deriazation):
            sys.exit()
            sys.exit(f"{deriazation} seems like smth went wrong")
        lastCut = 0
        res = None
    
        for i, segments in enumerate(deriazation):
                # duration = turn.end - turn.start
            cutFrame = math.floor(SAMPLERATE*segments["end"])
            cutFirst = math.floor(SAMPLERATE*segments["start"])
            print(f'{cutFirst}, {cutFrame}')
            # print(i)
            # print(self.reservedArray.size)


                    #check this
            # print("second phase")
            if(i==lastCut):
                self.reservedArray = np.concatenate([self.reservedArray, self.arrayOne[cutFirst:cutFrame]], axis=None)
            else:
                # print(self.reservedArray)

                # no longer sending data, just call it and make it check the queue if it is empty or nah before run
                if self.reservedArray.size < SAMPLERATE:
                    return None
                print("running.....")
                self.vadCut.append(self.reservedArray)
                if not self.lock.locked() and self.vadCut:
                    asyncio.create_task(
                        self.transcribe(ws)
                    )
                self.reservedArray = np.array([], dtype=np.float32)

                # res = self.chunk_validator(self.reservedArray)
                self.reservedArray = np.concatenate([self.reservedArray, self.arrayOne[cutFirst:cutFrame]], axis=None)
                
                # print(res)
                print("done transcribe")
        self.arrayOne = np.ndarray([], dtype=np.float32)
        return res

    async def transcribe(self, ws:WebSocket):
        # print("fucking fenomenal")

        async with self.lock:

            segment, result = self.modules.transcribe(
                self.vadCut.popleft(), 
                beam_size=5, 
                vad_filter=False, 
                condition_on_previous_text=False,   # very important for streaming
                word_timestamps=False,      # set True only if you need them (slows it down)
                temperature=0.0,)

            retData = ""
            for segment in segment:
                retData+=segment.text
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
            await ws.send_json(
                {"text": retData}
            )
            
        