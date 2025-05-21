from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
model = load_model('ethnicity_model.h5')

# Define the class labels in the order your model was trained on
ethnicity_classes = ['White', 'Black', 'Asian', 'Indian', 'Hispanic']

def predict_ethnicity(image_file, model):
    try:
        # Load and preprocess the image
        img = Image.open(image_file).convert('RGB')
        img = img.resize((48, 48))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Predict using model
        predictions = model.predict(img_array)[0]  # shape: (5,)
        percentages = [round(prob * 100, 2) for prob in predictions]

        results = list(zip(ethnicity_classes, percentages))

        return results

    except Exception as e:
        print(f"Prediction Error: {e}")
        return None

