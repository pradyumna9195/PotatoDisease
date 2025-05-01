from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import io
import base64
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)

# Load your trained model
model = load_model('models/model_1.h5')

# Define the IMAGE_SIZE
IMAGE_SIZE = 256

# Define a mapping of class indices to plant names and diseases
class_names = [
    'Potato__Healthy',
    'Potato__Early_blight',
    'Potato__Late_blight',
    # Add more classes as needed
]

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' in request.files:
        # Handle file upload
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        # Read and preprocess the image
        img = Image.open(io.BytesIO(file.read()))
    
    elif 'image_base64' in request.json:
        # Handle Base64-encoded image
        image_base64 = request.json['image_base64']
        image_data = base64.b64decode(image_base64)
        img = Image.open(io.BytesIO(image_data))
    
    else:
        return jsonify({'error': 'No image provided'}), 400

    # Resize to your model's expected input size
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize if needed

    # Make prediction
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_names[predicted_class_index]
    confidence = float(np.max(predictions)) * 100  # Convert to percentage

    # Create the response to match your required format
    response = {
        'Actual': predicted_class_name.split('__')[0],
        'Predicted': predicted_class_name,
        'Confidence': f"{confidence:.2f}%"
    }

    return jsonify(response)

# Run the application
if __name__ == '__main__':
    app.run(debug=True)