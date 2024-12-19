from flask import Flask, request
import os


app = Flask("Measurement server")

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file uploaded', 400
    
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    
    if not file.filename.lower().endswith('.png'):
        return 'File must be PNG', 400
    
    # save the file
    uploads_dir = 'uploads'
    os.makedirs(uploads_dir, exist_ok=True)
    file.save(os.path.join(uploads_dir, file.filename))
    
    return 'File uploaded successfully'


    



if __name__ == '__main__':
    app.run()