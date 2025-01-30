import os
import subprocess
from flask import Flask, render_template, request, redirect, url_for, send_file, flash
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import whisper
from gtts import gTTS
from dotenv import load_dotenv
import logging
from flask import jsonify
from flask import send_from_directory

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = "your_secret_key"

# Configure upload and processed directories
UPLOAD_FOLDER = "uploads"
PROCESSED_FOLDER = "processed"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# Allowed video file extensions
ALLOWED_EXTENSIONS = {"mp4", "mkv", "avi", "mov"}

# Set up logging for debugging
logging.basicConfig(level=logging.DEBUG)

# Check if a file has an allowed extension
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/")
def index():
    return render_template("frontend.html")

@app.route('/processed/<filename>')
def send_processed_file(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route("/upload", methods=["POST"])
def upload():
    try:
        if "video" not in request.files:
            flash("No file part in the request.")
            return jsonify({"error": "No file uploaded"}), 400

        
        video_file = request.files["video"]
        language = request.form.get("language")

        if video_file.filename == "":
            flash("No file selected.")
            return jsonify({"error": "No file selected"}), 400

        if video_file and allowed_file(video_file.filename):
            # Secure the uploaded file name
            filename = secure_filename(video_file.filename)
            video_path = os.path.join(UPLOAD_FOLDER, filename)
            video_file.save(video_path)
            logging.info(f"Video uploaded and saved at {video_path}.")

            # Start processing the video
            try:
                # Step 1: Extract audio from the video
                audio_path = os.path.join(PROCESSED_FOLDER, "extracted_audio.m4a")
                extract_audio(video_path, audio_path)
                logging.info("Audio extraction complete.")

                # Step 2: Transcribe the extracted audio
                transcribed_text_path = os.path.join(PROCESSED_FOLDER, "transcribed_text.txt")
                transcribe_audio(audio_path, transcribed_text_path)
                logging.info("Audio transcription complete.")

                # Step 2.5: Generate Summary
                summary_text_path = os.path.join(PROCESSED_FOLDER, "summary_text.txt")
                summarize_text(transcribed_text_path, summary_text_path)
                logging.info("Summary generation complete.")


                # Step 3: Translate the transcribed text
                translated_text_path = os.path.join(PROCESSED_FOLDER, "translated_text.txt")
                translate_text(transcribed_text_path, translated_text_path, language)
                logging.info("Text translation complete.")

                # Step 4: Generate translated audio from text
                translated_audio_path = os.path.join(PROCESSED_FOLDER, "translated_audio.mp3")
                generate_audio(translated_text_path, translated_audio_path, language)
                logging.info("Audio generation complete.")

                # Step 5: Mix original and translated audio
                mixed_audio_path = os.path.join(PROCESSED_FOLDER, "mixed_audio.mp3")
                mix_audio(audio_path, translated_audio_path, mixed_audio_path)
                logging.info("Audio mixing complete.")

                # Step 6: Replace audio in the video
                output_video_path = os.path.join(PROCESSED_FOLDER, f"translated_video_{language}.mp4")
                replace_audio_in_video(video_path, mixed_audio_path, output_video_path)
                logging.info(f"Video processing complete. Final video saved at {output_video_path}.")

                with open(summary_text_path, "r") as file:
                   summary_text = file.read()

                return jsonify({
                "videoUrl": url_for('send_processed_file', filename=f'translated_video_{language}.mp4'),
                "summary": summary_text
            })
            except Exception as e:
                return jsonify({"error": str(e)}), 500  
        else:
            return jsonify({"error": "Invalid file type"}), 400

    except Exception as e:
       return jsonify({"error": str(e)}), 500



# Helper Functions
def extract_audio(video_path, output_audio_path):
    logging.info(f"Extracting audio from {video_path} to {output_audio_path}.")
    command = ["ffmpeg", "-i", video_path, "-vn", "-acodec", "copy", output_audio_path]
    subprocess.run(command, check=True)

def transcribe_audio(audio_file, output_file):
    logging.info(f"Transcribing audio from {audio_file}.")
    model = whisper.load_model("small")
    result = model.transcribe(audio_file, language="en", temperature=0.7)
    transcribed_text = result["text"]
    with open(output_file, "w") as file:
        file.write(transcribed_text)

def translate_text(input_file, output_file, language):
    logging.info(f"Translating text from {input_file} to {language}.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts.prompt import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_template = PromptTemplate(
        input_variables=["language", "text"],
        template="""Translate the following text to {language}:\n\n{text}"""
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    chain = prompt_template | llm | StrOutputParser()

    with open(input_file, "r") as file:
        text_to_translate = file.read()
    res = chain.invoke({"language": language, "text": text_to_translate})

    with open(output_file, "w") as file:
        file.write(res)

def summarize_text(input_file, output_file):
    logging.info(f"Generating summary for {input_file}.")
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.prompts.prompt import PromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n\n{text}"
    )
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        api_key=os.getenv("GOOGLE_API_KEY")
    )
    chain = prompt_template | llm | StrOutputParser()

    with open(input_file, "r") as file:
        text_to_summarize = file.read()
    summary = chain.invoke({"text": text_to_summarize})

    with open(output_file, "w") as file:
        file.write(summary)


def generate_audio(input_file, output_file, language):
    logging.info(f"Generating audio in {language} from text file {input_file}.")
    with open(input_file, "r") as file:
        text = file.read()
    tts = gTTS(text=text, lang=language, slow=False)
    tts.save(output_file)

from pydub import AudioSegment

def mix_audio(original_audio_path, translated_audio_path, output_mixed_audio_path):
    """
    Mixes the original and translated audio, reducing the volume of the original audio.
    """
    original_audio = AudioSegment.from_file(original_audio_path)
    translated_audio = AudioSegment.from_file(translated_audio_path)

    # Reduce the volume of the original audio (optional, set to -120 dB to mute it entirely)
    original_audio = original_audio - 30  # Lower volume by 30 dB

    # Mix translated audio over the original (replace original if muted)
    mixed_audio = translated_audio.overlay(original_audio)

    # Export the mixed audio
    mixed_audio.export(output_mixed_audio_path, format="mp3")


def replace_audio_in_video(video_path, new_audio_path, output_video_path):
    """
    Replaces the audio of the video with the new audio.
    """
    command = [
        "ffmpeg", "-i", video_path, "-i", new_audio_path,
        "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
        "-shortest", output_video_path
    ]
    subprocess.run(command, check=True)

@app.route("/download/<filename>")
def download_video(filename):
    return send_file(os.path.join(PROCESSED_FOLDER, filename))

if __name__ == "__main__":
    app.run(debug=True)



