import os
import time

# Placeholder for speech recognition and translation libraries
# Implement real imports and methods when running in a compatible environment
try:
    import speech_recognition as sr
except ModuleNotFoundError:
    print("speech_recognition module not found. Ensure it is installed in your local environment.")
    sr = None

try:
    from googletrans import Translator
except ModuleNotFoundError:
    print("googletrans module not found. Ensure it is installed in your local environment.")
    Translator = None

# Define the active time period for translation (9:30 PM to 10:00 PM)
ACTIVE_START_HOUR = 17
ACTIVE_START_MINUTE = 56
ACTIVE_END_HOUR = 18
ACTIVE_END_MINUTE = 30

def is_active_period():
    """
    Checks if the current time is within the active time period.
    """
    current_time = time.localtime()
    start_time = time.struct_time((current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                                   ACTIVE_START_HOUR, ACTIVE_START_MINUTE, 0,
                                   current_time.tm_wday, current_time.tm_yday, current_time.tm_isdst))
    end_time = time.struct_time((current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                                 ACTIVE_END_HOUR, ACTIVE_END_MINUTE, 0,
                                 current_time.tm_wday, current_time.tm_yday, current_time.tm_isdst))
    return start_time <= current_time <= end_time

def recognize_speech():
    """
    Captures audio from the microphone and transcribes it to English text.
    Prompts user to repeat if audio is unclear.
    Returns the transcribed text.
    """
    if not sr:
        return "Error: Speech recognition module unavailable."

    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=1)  # Dynamically adjust to noise level
        print("Listening... Speak now!")
        try:
            # Capture audio
            audio = recognizer.listen(source, timeout=20, phrase_time_limit=20)
            print("Processing...")

            # Recognize speech using Google Web Speech API
            text = recognizer.recognize_google(audio, language='en-US')
            print(f"You said: {text}")
            return text

        except sr.WaitTimeoutError:
            print("Listening timed out. Please ensure you start speaking within the given time.")
            return "RETRY"
        except sr.UnknownValueError:
            print("Sorry, I couldn't understand the audio. Please try again.")
            return "RETRY"
        except sr.RequestError as e:
            print(f"Service error: {e}")
            return None
        except Exception as ex:
            print(f"An error occurred: {ex}")
            return None

def translate_to_hindi(text):
    """
    Translates English text to Hindi.
    """
    if not Translator:
        return "Error: Translation module unavailable."

    translator = Translator()
    try:
        translated = translator.translate(text, src='en', dest='hi')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

if __name__ == "__main__":
    if not is_active_period():
        print("Taking rest, see you tomorrow!")
    else:
        while True:
            result = recognize_speech()
            if result == "RETRY":
                continue
            elif result:
                hindi_translation = translate_to_hindi(result)
                if hindi_translation:
                    print(f"Translation in Hindi: {hindi_translation}")
                else:
                    print("Failed to translate the text.")
                break
            else:
                print("Exiting due to an error.")
                break
