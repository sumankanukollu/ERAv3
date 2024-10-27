from flask import Flask, render_template, request, jsonify, url_for
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_animal_image/<animal>')
def get_animal_image(animal):
    animal_images = {
        'cat': 'images/cat.jpg',
        'dog': 'images/dog.jpg',
        'elephant': 'images/elephant.jpg'
    }
    image_path = animal_images.get(animal, '')
    if image_path:
        return url_for('static', filename=image_path)
    return ''

@app.route('/upload_file', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    file_details = {
        'name': file.filename,
        'size': len(file.read()),
        'type': file.content_type
    }
    return jsonify(file_details)

if __name__ == '__main__':
    app.run(debug=True)
