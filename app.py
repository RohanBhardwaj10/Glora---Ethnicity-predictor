#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from predict import predict_ethnicity 

app = Flask(__name__)
model = load_model('ethnicity_model.h5')  #

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No image uploaded", 400

    image_file = request.files['image']

    if image_file.filename == '':
        return "No selected file", 400

    results = predict_ethnicity(image_file, model)

    if results is None:
        return "Error during prediction", 500

    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)


# In[ ]:




