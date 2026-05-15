import whisper
import os
import subprocess
from torch import Tensor
from numpy import ndarray
from faster_whisper import WhisperModel

class MainFunction:
    #future dev: timelapse to run
    #better version choosing
    def __init__(self, model):
        # self.SOURCE = "testFiles\\"
        self.MODEL = "large-v3-turbo"
        # self.ALLMODELS = ["tiny", "base", "small", "medium", "large", "turbo"]
        # if model in self.ALLMODELS:
        print("Model found, using "+model)
        # cuda if nvidia
        self.modules = WhisperModel(self.MODEL, device="cpu", compute_type="float32")
        # else:
            # print(f"Model not found, using turbo as default")
            # self.modules = whisper.load_model("base")
    
    def transcribe(self, wav_data: ndarray) -> str:

        returnedOutputs={}
        # for theFile in wav_16_file:
            # wav_file=self.SOURCE+theFile

            # if os.path.exists(wav_file):
            #     print("----------------------------")
            #     print("File found ✅")
            # else:
            #     print("----------------------------")
            #     print("File NOT found ❌")
            #     return ""
            
            # wav_file = self.checkHertz(wav_file)

        segment, result = self.modules.transcribe(
            wav_data, 
            beam_size=5, 
            vad_filter=False, 
            condition_on_previous_text=False,   # very important for streaming
            word_timestamps=False,      # set True only if you need them (slows it down)
            temperature=0.0,)

        retData = ""
        for segment in segment:
            retData+=segment.text
            # print("[%.2fs -> %.2fs] %s" % (segment.start, segment.end, segment.text))
        return retData
    
    
# data = MainFunction("1")
# data.transcribe(["Recording.wav"])
