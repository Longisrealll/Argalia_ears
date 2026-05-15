import whisper
import os
import subprocess
from torch import Tensor
import numpy as np
import sounddevice as sd
import asyncio

THEPCM = np.array([], dtype=np.float32)
count = 0
overall = True

async def main():
    global THEPCM, count, overall

    model = whisper.load_model("base")
    queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time, status):
        global THEPCM, count

        chunk = indata.squeeze().astype(np.float32)

        THEPCM = np.concatenate([THEPCM, chunk], axis=0)
        count += 1

        loop.call_soon_threadsafe(queue.put_nowait, chunk)

    stream = sd.InputStream(
        callback=callback,
        samplerate=16000,
        channels=1,
        dtype="float32",
        blocksize=16000
    )
    stream.start()

    while overall:
        _ = await queue.get()

        if count >= 10:
            audio = THEPCM.copy()
            print(THEPCM.size)

            if len(audio) > 0:
                result = model.transcribe(audio)
                print(result)

            THEPCM = np.array([], dtype=np.float32)
            count = 0

asyncio.run(main())