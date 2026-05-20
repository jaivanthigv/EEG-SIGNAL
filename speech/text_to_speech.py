from gtts import gTTS
from playsound import playsound
import tempfile
import os
import uuid

def speak(text):
    try:
        filename = f"{uuid.uuid4()}.mp3"
        filepath = os.path.join(tempfile.gettempdir(), filename)

        tts = gTTS(text=text, lang='en')
        tts.save(filepath)
        playsound(filepath)

        os.remove(filepath)  # clean up
    except Exception as e:
        print("Speech error:", e)
