import whisper


def audio_transcription(audio_file):
    # Load the "base" model from the whisper library
    model = whisper.load_model("base")

    # Perform audio transcription using the loaded model
    # fp16=False specifies not to use half-precision (FP16) floating-point format
    result = model.transcribe(audio_file, fp16=False)

    # Retrieve the transcribed text from the result dictionary
    transcribed_text = result["text"]

    # Return the transcribed text
    return transcribed_text
