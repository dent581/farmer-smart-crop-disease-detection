# 🌿 FarmAdvisor AI — Crop Disease Detection System

A complete production-ready Flask web application for AI-powered crop disease detection and agricultural advisory.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create demo CNN model (no dataset needed for testing)
python train_image_model.py --demo

# 3. Train NLP advisory model
python train_text_model.py

# 4. Launch the application
python app.py

# Visit: http://127.0.0.1:5000
```

## 📁 Project Structure

```
farmer_advisory/
├── app.py                   # Main Flask application
├── database.py              # SQLite persistence layer
├── train_image_model.py     # CNN training script
├── train_text_model.py      # NLP model training script
├── requirements.txt
├── models/
│   ├── plant_disease_model.h5   # CNN model (after training)
│   ├── text_model.pkl           # NLP model (after training)
│   ├── class_names.json         # 21 disease class labels
│   └── text_labels.json
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   └── uploads/             # Uploaded leaf images
├── templates/
│   ├── index.html           # Home Dashboard
│   ├── upload.html          # Image Scan page
│   ├── query.html           # Ask AI page
│   └── history.html         # History page
└── farmer_advisory.db       # SQLite database
```

## 🧠 Training with Real Dataset

Download the **PlantVillage dataset** from Kaggle, then:

```bash
python train_image_model.py --data_dir /path/to/plantvillage --epochs 25
```

The script uses **MobileNetV2 transfer learning** with two phases:
1. Train classifier head (frozen backbone)
2. Fine-tune top 40 layers

## 🌾 Supported Disease Classes (21)

| Crop    | Diseases |
|---------|----------|
| Apple   | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| Corn    | Gray Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| Potato  | Early Blight, Late Blight, Healthy |
| Tomato  | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, TYLCV, Mosaic Virus, Healthy |

## 🔧 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Home dashboard |
| `/scan` | GET | Image scan page |
| `/scan` | POST | Upload image → JSON prediction |
| `/ask` | GET | Ask AI page |
| `/ask` | POST | Query text → JSON advisory |
| `/history` | GET | History page |
| `/api/history/clear` | POST | Clear all records |
| `/api/status` | GET | Model status JSON |
