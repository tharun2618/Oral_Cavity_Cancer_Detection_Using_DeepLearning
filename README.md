# Oral Cancer Detection

A Flask web application that analyzes oral cavity images and predicts whether they show signs of oral cancer. Each uploaded image is evaluated by **three independent methods** and the results are displayed together:

| Method | Type | Details |
|--------|------|---------|
| Groq LLM | Vision language model | `meta-llama/llama-4-scout-17b-16e-instruct` analyzes the image and returns a JSON diagnosis |
| EfficientNet / VGG19 | Custom-trained CNN | Fine-tuned on a balanced oral-cancer dataset; reports cancer confidence (%) |
| DenseNet121 | Custom-trained CNN | Fine-tuned with data augmentation and two-stage training; reports cancer confidence (%) |

The application renders the uploaded image in grayscale and HSV-thresholded views, shows the final diagnosis, and plots the two model confidence scores side by side for comparison.

## Features

- User registration and login (MySQL-backed, session-based authentication)
- Image upload with preprocessing previews (grayscale, thresholded)
- Multi-model diagnosis via LLM and two custom CNNs
- Accuracy comparison chart generated per prediction
- Upload history is preserved on disk in `static/docs/`

## Tech Stack

- **Backend:** Python 3.9+, Flask, MySQL (`mysql-connector-python`)
- **Deep learning:** TensorFlow 2.16, Keras (DenseNet121, EfficientNet/VGG19)
- **Image processing:** OpenCV, Pillow, NumPy
- **Vision LLM:** Groq API (Llama 4 Scout)

## Project Structure

```
Oral_Cancer_Detection/
├── app.py                  # Flask application (routes, model inference)
├── supportcode.py          # Accuracy plot helper
├── oraldisease_2026.sql    # Database schema + seed user
├── requirements.txt        # Python dependencies
├── templates/              # HTML templates (login, register, upload, etc.)
├── static/                 # CSS, JS, and runtime-generated image output
├── training_scripts/       # Colab notebook used to train the models
├── models/                 # Trained .h5 model files (not tracked in git)
└── Dataset/                # Training dataset (not tracked in git)
```

## Prerequisites

- Python 3.9 or newer
- MySQL server (8.x recommended)
- A [Groq API key](https://console.groq.com/keys)

## Setup

### 1. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If the install fails, the key pinned packages are `tensorflow==2.16.1`, `numpy==1.26.4`, and `mysql-connector-python` — install them individually, then install the rest.

### 3. Set up the database

1. Start your MySQL server.
2. Create a database named `oraldisease_2026`:
   ```sql
   CREATE DATABASE oraldisease_2026;
   ```
3. Import the schema:
   ```bash
   mysql -u root oraldisease_2026 < oraldisease_2026.sql
   ```
   (or import `oraldisease_2026.sql` through phpMyAdmin / MySQL Workbench).

4. If your MySQL credentials differ, update the connection settings at the top of `app.py`.

The seed user `user@user.com` / `user` is included for quick testing.

### 4. Add the trained model files

Place the two trained models in the `models/` directory:

- `models/EfficientNetorVGG19_Oral_Cancer_BALANCED.h5`
- `models/DenseNet_Oral_Cancer_BALANCED.h5`

These files are excluded from git (see `.gitignore`) — they are trained artifacts and must be downloaded separately. Without them the app still runs, but model predictions fall back to `0.0%`. The training notebook in `training_scripts/` can be used to reproduce them.

### 5. Configure the Groq API key

Set the `GROQ_API_KEY` environment variable:

```bash
# macOS / Linux
export GROQ_API_KEY="your-groq-api-key"
# Windows (PowerShell)
$env:GROQ_API_KEY = "your-groq-api-key"
# Windows (cmd)
set GROQ_API_KEY=your-groq-api-key
```

The app reads the key from this variable; it is no longer hardcoded in the source.

### 6. Run the application

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

## Usage

1. Register an account (or log in with the seeded `user@user.com` / `user`).
2. Upload an oral cavity image.
3. Review the preprocessed views (grayscale, thresholded) and the diagnosis.
4. Compare the accuracy scores of the two CNN models on the generated chart.

## Database Schema

The `oraldisease_2026_user` table stores registered users:

| Column   | Type         | Description            |
|----------|--------------|------------------------|
| `id`     | int (PK)     | Auto-increment ID      |
| `uid`    | varchar(255) | Unique user identifier |
| `name`   | varchar(255) | Full name              |
| `email`  | varchar(255) | Login email            |
| `password` | varchar(255) | Login password       |
| `phone`  | varchar(255) | Contact number         |

## Disclaimer

This project is for **educational and research purposes only**. It is not a medical device and must not be used as a substitute for professional medical diagnosis. Always consult a qualified healthcare provider for any health concerns.

## License

[MIT](LICENSE)
