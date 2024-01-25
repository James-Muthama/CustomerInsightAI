import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from CustomerInsightAI.Audio_file_to_text.audio_transcriber import audio_transcription
from CustomerInsightAI.AI_categoriser.categoriser import categorize

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'C:/Users/James Muthama/PycharmProjects/ABSAproject/CustomerInsightAI/uploads'

ALLOWED_EXTENSIONS = {'mp3'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})

        uploaded_file = request.files['file']

        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'})

        if uploaded_file and allowed_file(uploaded_file.filename):
            # Save the uploaded MP3 file to a temporary file
            filename = uploaded_file.filename
            file_path = (os.path.join(app.config['UPLOAD_FOLDER'], filename))
            uploaded_file.save(file_path)

            # Perform audio transcription using the file path
            customer_care_text = audio_transcription(file_path)

            print(customer_care_text)

            # Perform categorization
            categories, descriptions = categorize(customer_care_text)

            # Return the results as JSON
            if not categories or not descriptions:
                return jsonify({'categories': "Unfortunately I was unable to classify the audio you put in",
                                'descriptions': "Unfortunately I was unable to classify the audio you put in"})
            else:
                return jsonify({'categories': categories, 'descriptions': descriptions})
        else:
            return jsonify({'error': 'Invalid file type. Please upload an MP3 file'})
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
