from flask import Flask, render_template, request, jsonify
import time
from googletrans import Translator

app = Flask(__name__)

# Define the active time period for translation (9:30 PM to 10:00 PM)
ACTIVE_START_HOUR = 21
ACTIVE_START_MINUTE = 30
ACTIVE_END_HOUR = 22
ACTIVE_END_MINUTE = 00

def is_active_period():
    current_time = time.localtime()
    start_time = time.struct_time((current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                                   ACTIVE_START_HOUR, ACTIVE_START_MINUTE, 0,
                                   current_time.tm_wday, current_time.tm_yday, current_time.tm_isdst))
    end_time = time.struct_time((current_time.tm_year, current_time.tm_mon, current_time.tm_mday,
                                 ACTIVE_END_HOUR, ACTIVE_END_MINUTE, 0,
                                 current_time.tm_wday, current_time.tm_yday, current_time.tm_isdst))
    return start_time <= current_time <= end_time

def translate_to_hindi(text):
    translator = Translator()
    try:
        translated = translator.translate(text, src='en', dest='hi')
        return translated.text
    except Exception as e:
        print(f"Translation error: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translate", methods=["POST"])
def translate():
    if not is_active_period():
        return jsonify({"error": "The translation service is only available between 9:30 PM and 10:00 PM."})
    
    text = request.form.get("text")
    if not text:
        return jsonify({"error": "No input text provided."})
    
    hindi_translation = translate_to_hindi(text)
    if "Error" in hindi_translation:
        return jsonify({"error": hindi_translation})
    
    return jsonify({"translation": hindi_translation})


if __name__ == "__main__":
    app.run(debug=True)
