import pyttsx3
import queue
import threading


speech_engine = pyttsx3.init()
speech_engine.setProperty('rate', 175)  # Adjust speech rate
speech_engine.setProperty('volume', 2)  # Adjust volume
speech_queue = queue.Queue()

def speak_thread():
    while True:
        text = speech_queue.get()
        if text == "exit":
            break
        # Say the text
        speech_engine.say(text)
        speech_engine.runAndWait()

thread = threading.Thread(target=speak_thread, daemon=True)
thread.start()

def speak(text):
    speech_queue.put(text)
