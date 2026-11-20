import cv2
from flask import Flask, redirect, render_template, request, session, url_for, jsonify
import mysql.connector, random, string, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from sklearn.metrics import *
import subprocess
import tensorflow as tf
from PIL import Image
from groq import Groq
from PIL import Image
import json
import base64
import io
from tensorflow.keras.applications import VGG19, DenseNet121
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess_input



app = Flask(__name__)
app.secret_key = "Qazwsx@123"


link = mysql.connector.connect(
    host = 'localhost',
    user = 'root',
    password = '',
    database = 'oraldisease_2026'
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


@app.after_request
def add_header(response):

  response.cache_control.no_store = True
  return response



@app.route('/')
def index():

  return render_template('index.html')



@app.route('/about')
def about():

  return render_template('about.html')




@app.route('/login', methods=['GET', 'POST'])
def login():

  if 'user' in session:
    return redirect(url_for('upload'))

  if request.method == "GET":
    return render_template('login.html')

  else:
    cursor = link.cursor()
    try:
      email = request.form["email"]
      password = request.form["password"]
      cursor.execute("SELECT * FROM oraldisease_2026_user WHERE email = '"+email+"' AND password = '"+password+"'")
      user = cursor.fetchone()
      if user:
        session['user'] = user[3]
        session['username'] = user[2]
        return redirect(url_for('upload'))
      else:
        return render_template('login.html', error='Invalid email or password')

    except Exception as e:
      error = e
      return render_template('login.html', error=error)

    finally:
        cursor.close()




@app.route('/register', methods=['GET', 'POST'])
def register():

  if 'user' in session:
    return redirect(url_for('upload'))

  if request.method == "GET":
    return render_template('register.html')

  else:
    cursor = link.cursor()
    try:
      name = request.form["name"]
      email = request.form["email"]
      password = request.form["password"]
      phone = request.form["phone"]
      uid = 'uid_'+''.join(random.choices(string.ascii_letters + string.digits, k=10))
      cursor.execute("SELECT * FROM oraldisease_2026_user WHERE email = %s", (email,))
      user = cursor.fetchone()

      if user:
        return render_template('register.html', exists='Email already exists')
      else:
        cursor.execute("INSERT INTO oraldisease_2026_user (uid, name, email, password, phone) VALUES ('"+uid+"', '"+name+"', '"+email+"', '"+password+"', '"+phone+"')")
        link.commit()
        return render_template('register.html', success='Registration successful')

    except Exception as e:
      error = e
      return render_template('register.html', error=error)

    finally:
        cursor.close()

@app.route('/upload', methods=['GET', 'POST'])
def upload():

  if 'user' not in session:
    return redirect(url_for('login'))

  if request.method == "GET":
    return render_template('upload.html')

  else:
    cursor = link.cursor()
    image = request.files["image"]
    imagepath = os.path.join(os.path.dirname(os.path.abspath(__file__)) + '\\static\\docs', image.filename)
    image.save(imagepath)
    ci=cv2.imread(imagepath)
    val=os.stat(imagepath).st_size
    gray = cv2.cvtColor(ci, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("static/Grayscale/"+image.filename,gray)
    thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    cv2.imwrite("static/Threshold/"+image.filename,thresh)
    thresh = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    lower_green = np.array([34, 177, 76])
    upper_green = np.array([255, 255, 255])
    hsv_img = cv2.cvtColor(ci, cv2.COLOR_BGR2HSV)
    binary = cv2.inRange(hsv_img, lower_green, upper_green)
    cv2.imwrite("static/Binary/"+image.filename,gray)

    image_path = imagepath
    image_base64 = encode_image(image_path)

    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        return render_template('upload.html', error='GROQ_API_KEY environment variable is not set')
    client = Groq(api_key=groq_api_key)
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "You are an expert in medical imaging analysis.\n\n"
                            "Analyze the given oral cavity image carefully and classify it as one of the following:\n"
                            "- 'Cancer' if the image shows signs of oral cancer.\n"
                            "- 'Non-cancer' if the image does not show signs of oral cancer.\n\n"
                            "Focus on details such as:\n"
                            "- Abnormal lesions, discolorations, or growths\n"
                            "- Texture irregularities or tissue inflammation\n"
                            "- Presence of malignant-appearing patterns\n\n"
                            "Return ONLY the result in the exact JSON format below:\n"
                            "{\"diagnosis\": \"Cancer\"} or {\"diagnosis\": \"Non-cancer\"}\n\n"
                            "Do not include any explanation, text, or comments. Only return the JSON."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}",
                        },
                    },
                ],
            }
        ],
        model="meta-llama/llama-4-scout-17b-16e-instruct",
    )

    response_text = chat_completion.choices[0].message.content
    cleaned_data = json.loads(response_text[response_text.find("{"):response_text.rfind("}") + 1])

    acc = get_efficientnet_prediction(imagepath)
    acc1 = get_densenet_prediction(imagepath)
    if acc1 > 50:
        op = "Cancer"
    else:
        op = "Non-Cancer"

    try:
      acc_float = float(acc) if acc != "Not Available" else 0
      acc1_float = float(acc1) if acc1 != "Not Available" else 0

      labels = ['VGG19 Accuracy', 'DenseNet Accuracy']

      accuracies = [acc_float, acc1_float]
      plt.figure(figsize=(8, 6))
      accuracies = [acc_float, acc1_float]
      plt.figure(figsize=(8, 6))
      plt.bar(labels, accuracies, color=['blue', 'green'])
      plt.xlabel('Model')
      plt.ylabel('Accuracy (%)')
      plt.title('Comparison of Model Accuracies')
      plt.ylim(0, 100)
      plt.savefig(os.path.join('static', 'comparison_graph.png'))
      plt.close()

    except Exception as e:
        print(f"Error generating plot: {e}")
        plot_url = None
    return render_template('upload.html', success='Upload successful', image=image.filename,op=op,acc=acc,acc1=acc1)

@app.route('/logout')
def logout():

    session.pop('user', None)
    session.pop('username', None)
    return redirect(url_for('index'))


def get_efficientnet_prediction(image_path):
    try:
        model_path = 'models/EfficientNetorVGG19_Oral_Cancer_BALANCED.h5'

        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Returning 0.0")
            return 0.0

        model = load_model(model_path)

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)

        prediction = model.predict(img_array_expanded)

        cancer_confidence = prediction[0][0] * 100

        return cancer_confidence
    except Exception as e:
        print(f"Error in EfficientNet prediction: {e}")
        return 0.0


def get_densenet_prediction(image_path):
    try:
        model_path = 'models/DenseNet_Oral_Cancer_BALANCED.h5'
        if not os.path.exists(model_path):
            print(f"Warning: Model file not found at {model_path}. Returning 0.0")
            return 0.0

        model = load_model(model_path)

        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array_expanded = np.expand_dims(img_array, axis=0)

        img_preprocessed = densenet_preprocess_input(img_array_expanded)

        prediction = model.predict(img_preprocessed)

        cancer_confidence = prediction[0][0] * 100

        return cancer_confidence
    except Exception as e:
        print(f"Error in DenseNet prediction: {e}")
        return 0.0


if __name__ == '__main__':
    app.run(debug=True)
