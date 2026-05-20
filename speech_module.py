import pyttsx3
import threading

def speak_text(text):
    """Speak text using system voice (optional backup)."""
    try:
        def run():
            engine = pyttsx3.init()
            voices = engine.getProperty('voices')

            # Try to select female voice if available
            for v in voices:
                if "female" in v.name.lower() or "zira" in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break

            engine.setProperty('rate', 150)
            engine.setProperty('volume', 1)

            engine.say(str(text))
            engine.runAndWait()

        threading.Thread(target=run).start()

    except Exception as e:
        print("Speech Error:", e)


# 🚫 DISABLED (to avoid PyAudio error)
def recognize_speech():
    return "Mic input disabled (use browser voice)"