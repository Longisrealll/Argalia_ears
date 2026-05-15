import numpy as np
import sounddevice as sd
import asyncio
import torch
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
import math
from collections import deque
from silero_vad import load_silero_vad, read_audio, get_speech_timestamps
from mainCall import MainFunction

# make a roller 5 secs, parse after 3 secs, take 4 secs if they are long enough, check for user and time each 4 secs
# 

class Transcriber:
    def __init__(self):
        # self.bufferOne = np.ndarray([], dtype=np.float32)
        self.arrayOne = np.ndarray([], dtype=np.float32)
        self.reservedArray = np.ndarray([], dtype=np.float32)
        self.model = load_silero_vad()
        
        self.rolling_buffer = deque(maxlen=3)
        self.theWhisper = MainFunction("base")
        self.current = "SPEAKER_00"
    async def main(self):
        self.counter = 0
        
        leQueu = asyncio.Queue()

        loop = asyncio.get_running_loop()
        def callback(indata, frames, time, status):
            if status:
                print(status)
            chunk = indata.copy()

            self.rolling_buffer.append(chunk)
            self.counter+=1

            loop.call_soon_threadsafe(
                leQueu.put_nowait, chunk
            )

        theAudio = sd.InputStream(callback=callback, samplerate=16000, channels=1, dtype='float32', blocksize=16000)
        theAudio.start()

        async def runningAudio():
            while True:
                inputs = await leQueu.get()
                if(self.counter >= 3):
                    # if(len(self.rolling_buffer)==3):
                    for i in range(3):
                        self.arrayOne = np.concatenate([self.arrayOne, list(self.rolling_buffer[i])], axis=None)
                # else:
                #     for i in range(4):
                #         self.arrayOne = np.concatenate([self.arrayOne, list(self.rolling_buffer[i])], axis=None)
                    finalData = self.checkingData(self.arrayOne)
                    self.arrayOne = np.array([], dtype=np.float32)
                    self.counter = 0
                if inputs is None:
                    break
                # self.arrayOne = np.append(self.arrayOne, inputs[:,0].astype(np.float32))
                # print(len(self.arrayOne))
            
            # theDict = currentData

        try:
            await asyncio.gather(
                runningAudio()
                # countdown()
            )

        except asyncio.CancelledError:
            print("Final data: ")
            print(self.reservedArray.size)
            theAudio.stop()
            theAudio.close()

        except KeyboardInterrupt:
            theAudio.stop()
            theAudio.close()

    def chunk_validator(self, revArr):
        if revArr.size < 16000:
            return None
        # waveform = torch.from_numpy(revArr).unsqueeze(0)
        # audio = np.array(revArr, dtype=np.float32).squeeze()
        transcribed = self.theWhisper.transcribe(revArr)
        self.reservedArray = np.array([], dtype=np.float32)
        return transcribed
        # pass to the whisper, before that take 5 secs off the reserved array to check in the pipeline

    def checkingData(self, theList):
        deriazation = get_speech_timestamps(
            theList,
            self.model,
            sampling_rate=16000,
            min_silence_duration_ms=100,
            return_seconds=True
        )
        lastCut = 0
        # dataFin = {}
        res = None
    
        for i, segments in enumerate(deriazation):
                # duration = turn.end - turn.start
            cutFrame = math.floor(16000*segments["end"])
            cutFirst = math.floor(16000*segments["start"])
            print(f'{cutFirst}, {cutFrame}')
            print(self.reservedArray.size)


                    #check this
            if(i==lastCut):
                self.reservedArray = np.concatenate([self.reservedArray, theList[cutFirst:cutFrame]], axis=None)
            else:
                print(self.reservedArray)
                res = self.chunk_validator(self.reservedArray)
                self.reservedArray = np.concatenate([self.reservedArray, theList[cutFirst:cutFrame]], axis=None)
                
                print(res)
                print("done transcribe")
        return res
        
data = Transcriber()
asyncio.run(data.main())