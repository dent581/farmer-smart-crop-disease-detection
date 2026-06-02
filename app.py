"""
app.py — Farmer Advisory System — Main Flask Application
=========================================================
Run: python app.py
"""

import os
import json
import uuid
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["SECRET_KEY"]        = "farmer-ai-secret-2024"
app.config["UPLOAD_FOLDER"]     = "static/uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024   # 16 MB
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png", "webp"}

# ──────────────────────────────────────────────────────────────────
# Global model state
# ──────────────────────────────────────────────────────────────────
cnn_model            = None
class_names: list    = []
text_model           = None
text_vectorizer      = None
text_label_encoder   = None
MODEL_LOADED         = False
TEXT_MODEL_LOADED    = False

# ──────────────────────────────────────────────────────────────────
# Disease knowledge base  (21 PlantVillage classes)
# ──────────────────────────────────────────────────────────────────
DISEASE_INFO = {
    "Apple___Apple_scab": {
        "display": "Apple Scab",
        "severity": "Moderate",
        "description": "Fungal disease causing olive-green to brown scab-like lesions on leaves and fruit.",
        "fertilizer": "Apply balanced NPK (10-10-10) and organic compost. Include magnesium and zinc to boost immunity.",
        "pesticide": "Use fungicides like Captan or Mancozeb every 7–10 days during wet weather. Ensure leaf coverage.",
        "irrigation": "Use drip irrigation. Avoid overhead sprinklers, water early in the day.",
        "prevention": [
            "Plant scab-resistant apple varieties.",
            "Rake and destroy fallen leaves in autumn.",
            "Prune trees to improve air circulation.",
            "Apply dormant oil spray in early spring.",
            "Maintain proper spacing between trees.",
            "Remove infected shoots immediately."
        ]
    },
    "Apple___Black_rot": {
        "display": "Apple Black Rot",
        "severity": "Severe",
        "description": "Botryosphaeria fungus causing circular leaf lesions and fruit rot with concentric rings.",
        "fertilizer": "Balanced fertilization with potassium to strengthen cell walls. Avoid excess nitrogen. Add organic matter.",
        "pesticide": "Apply Captan, Mancozeb, or Thiophanate-methyl from pink bud stage and repeat every 10–14 days.",
        "irrigation": "Ensure good drainage. Water early morning to allow leaves to dry.",
        "prevention": [
            "Remove mummified fruit.",
            "Prune dead or infected wood.",
            "Maintain tree vigor.",
            "Apply protective fungicides at bud break.",
            "Rotate apple varieties.",
            "Monitor during rainy periods."
        ]
    },
    "Apple___Cedar_apple_rust": {
        "display": "Cedar Apple Rust",
        "severity": "Moderate",
        "description": "Fungal disease requiring both apple and cedar hosts; causes bright orange lesions on leaves.",
        "fertilizer": "Use zinc and manganese-rich fertilizer. Avoid excessive nitrogen.",
        "pesticide": "Apply Myclobutanil or Propiconazole from pink bud through petal fall.",
        "irrigation": "Avoid wetting foliage; use drip irrigation.",
        "prevention": [
            "Remove nearby cedar/juniper plants.",
            "Plant rust-resistant apple varieties.",
            "Apply preventive fungicides at bud break.",
            "Remove infected galls on cedars.",
            "Prune trees for airflow.",
            "Maintain proper canopy spacing."
        ]
    },
    "Apple___healthy": {
        "display": "Healthy Apple",
        "severity": "None",
        "description": "Apple plant appears healthy with no disease.",
        "fertilizer": "Regular balanced NPK, compost, micronutrients as per soil test.",
        "pesticide": "Preventive copper spray before rainy season; scout regularly.",
        "irrigation": "Deep water 1–2 times per week, preferably drip irrigation.",
        "prevention": [
            "Monitor leaves and fruit regularly.",
            "Maintain pruning and canopy management.",
            "Keep field records for nutrients and pests.",
            "Annual soil tests to adjust fertilization.",
            "Inspect for early signs of disease."
        ]
    },
    "Blueberry___healthy": {
        "display": "Healthy Blueberry",
        "severity": "None",
        "description": "Blueberry plant appears healthy with no disease.",
        "fertilizer": "Apply acid-loving fertilizer (ammonium sulfate), mulch to maintain soil acidity, supplement Mg and B.",
        "pesticide": "No treatment needed; scout for pests and fungi.",
        "irrigation": "Maintain consistent soil moisture; drip irrigation preferred.",
        "prevention": [
            "Mulch to conserve moisture and control weeds.",
            "Annual pruning for airflow.",
            "Monitor for aphids and mites.",
            "Test soil pH regularly (ideal 4.5–5.5).",
            "Ensure adequate spacing."
        ]
    },
    "Cherry_(including_sour)___Powdery_mildew": {
        "display": "Cherry Powdery Mildew",
        "severity": "Moderate",
        "description": "White powdery fungal growth on leaves and stems.",
        "fertilizer": "Balanced NPK, avoid excess nitrogen. Add phosphorus and potassium.",
        "pesticide": "Sulfur-based fungicides or Myclobutanil during humid conditions.",
        "irrigation": "Avoid overhead watering; maintain moderate soil moisture.",
        "prevention": [
            "Prune for airflow and sunlight.",
            "Remove infected shoots.",
            "Space plants properly.",
            "Sanitize tools before pruning.",
            "Monitor in spring and early summer."
        ]
    },
    "Cherry_(including_sour)___healthy": {
        "display": "Healthy Cherry",
        "severity": "None",
        "description": "Cherry plant is healthy with no visible disease.",
        "fertilizer": "Balanced NPK and organic compost.",
        "pesticide": "No treatment required; preventive scouting.",
        "irrigation": "Deep watering, avoid wet foliage.",
        "prevention": [
            "Maintain tree canopy for airflow.",
            "Mulch to conserve moisture.",
            "Monitor for pests and diseases.",
            "Winter pruning to remove dead wood."
        ]
    },
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": {
        "display": "Gray Leaf Spot",
        "severity": "Moderate",
        "description": "Fungal disease causing rectangular lesions on leaves reducing photosynthesis.",
        "fertilizer": "Balanced NPK with emphasis on potassium. Avoid excessive nitrogen.",
        "pesticide": "Use fungicides like Azoxystrobin or Pyraclostrobin when lesions appear.",
        "irrigation": "Avoid overhead irrigation; water early morning.",
        "prevention": [
            "Rotate crops to prevent pathogen buildup.",
            "Remove and destroy crop residues.",
            "Use resistant maize varieties.",
            "Maintain proper plant spacing.",
            "Monitor during humid weather."
        ]
    },
    "Corn_(maize)___Common_rust_": {
        "display": "Common Rust",
        "severity": "Moderate",
        "description": "Reddish-brown pustules on leaves reducing yield.",
        "fertilizer": "Balanced NPK; potassium to strengthen leaves.",
        "pesticide": "Fungicides like Propiconazole or Mancozeb at first sign of pustules.",
        "irrigation": "Water early; avoid wetting leaves.",
        "prevention": [
            "Use rust-resistant varieties.",
            "Remove volunteer maize plants.",
            "Rotate crops.",
            "Ensure good airflow.",
            "Monitor for early infection."
        ]
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "display": "Northern Leaf Blight",
        "severity": "Severe",
        "description": "Large cigar-shaped lesions on leaves affecting photosynthesis.",
        "fertilizer": "Balanced NPK, avoid high nitrogen to reduce lush susceptible growth.",
        "pesticide": "Apply fungicides like Azoxystrobin or Tebuconazole early in disease onset.",
        "irrigation": "Avoid prolonged leaf wetness; drip irrigation preferred.",
        "prevention": [
            "Rotate crops to reduce pathogen buildup.",
            "Plant resistant hybrids.",
            "Remove infected debris.",
            "Monitor frequently during wet periods.",
            "Maintain proper plant density."
        ]
    },
    "Corn_(maize)___healthy": {
        "display": "Healthy Corn",
        "severity": "None",
        "description": "Corn plants appear healthy.",
        "fertilizer": "Apply balanced NPK based on soil test.",
        "pesticide": "No chemical treatment needed; scout regularly.",
        "irrigation": "Deep, infrequent watering; avoid wetting foliage.",
        "prevention": [
            "Maintain proper plant spacing.",
            "Rotate crops.",
            "Monitor for early signs of disease.",
            "Keep fields weed-free.",
            "Use certified seeds."
        ]
    },
    "Grape___Black_rot": {
        "display": "Grape Black Rot",
        "severity": "Severe",
        "description": "Fungal disease causing dark lesions on leaves and berries.",
        "fertilizer": "Balanced NPK with extra potassium to strengthen tissue.",
        "pesticide": "Apply Mancozeb or Captan regularly during wet periods.",
        "irrigation": "Avoid overhead irrigation; use drip to keep foliage dry.",
        "prevention": [
            "Remove mummified berries.",
            "Prune for airflow.",
            "Use resistant grape varieties.",
            "Sanitize tools after pruning.",
            "Monitor during rainy season."
        ]
    },
    "Grape___Esca_(Black_Measles)": {
        "display": "Esca (Black Measles)",
        "severity": "Moderate",
        "description": "Fungal disease causing dark spots on berries and leaf chlorosis.",
        "fertilizer": "Balanced NPK with focus on nitrogen and potassium; add compost.",
        "pesticide": "Apply Bordeaux mixture or copper fungicides preventively.",
        "irrigation": "Ensure good drainage; avoid waterlogging.",
        "prevention": [
            "Remove infected wood and berries.",
            "Prune to improve airflow.",
            "Use certified disease-free planting material.",
            "Monitor regularly."
        ]
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "display": "Grape Leaf Blight",
        "severity": "Moderate",
        "description": "Fungal leaf spots causing defoliation and reduced yield.",
        "fertilizer": "Balanced NPK; potassium improves leaf resistance.",
        "pesticide": "Apply fungicides such as Copper oxychloride or Mancozeb.",
        "irrigation": "Avoid overhead sprinklers; water early.",
        "prevention": [
            "Remove fallen leaves.",
            "Prune to allow sunlight and airflow.",
            "Monitor for early symptoms.",
            "Use disease-free planting material."
        ]
    },
    "Grape___healthy": {
        "display": "Healthy Grape",
        "severity": "None",
        "description": "Grape plants are healthy with no visible disease.",
        "fertilizer": "Apply balanced NPK; add compost or manure annually.",
        "pesticide": "No chemical treatment needed; monitor for pests.",
        "irrigation": "Deep watering 1–2 times per week; drip irrigation preferred.",
        "prevention": [
            "Regular pruning for airflow.",
            "Mulch to retain moisture.",
            "Monitor for early pests or disease.",
            "Remove debris from vineyard."
        ]
    },
    "Orange___Haunglongbing_(Citrus_greening)": {
        "display": "Citrus Greening",
        "severity": "Severe",
        "description": "Bacterial disease causing yellowing leaves, misshapen fruit, and reduced yield.",
        "fertilizer": "Balanced NPK; supplement with magnesium, zinc, and manganese. Foliar feeding helps nutrient uptake.",
        "pesticide": "Control psyllid vectors with insecticides like Imidacloprid. Apply copper-based bactericides preventively.",
        "irrigation": "Avoid water stress; maintain consistent soil moisture. Avoid wetting foliage.",
        "prevention": [
            "Remove and destroy infected trees.",
            "Use certified disease-free seedlings.",
            "Plant disease-tolerant citrus varieties.",
            "Monitor and control psyllid populations.",
            "Sanitize tools and equipment."
        ]
    },
    "Peach___Bacterial_spot": {
        "display": "Peach Bacterial Spot",
        "severity": "Moderate",
        "description": "Bacterial lesions on leaves and fruit causing defoliation and blemishes.",
        "fertilizer": "Balanced NPK; avoid excess nitrogen. Add calcium for tissue strength.",
        "pesticide": "Copper-based bactericides at bud break and pre-bloom stages.",
        "irrigation": "Water at soil level; avoid wetting leaves and fruit.",
        "prevention": [
            "Prune infected branches.",
            "Use resistant varieties.",
            "Sanitize pruning tools.",
            "Remove fallen debris.",
            "Maintain proper spacing for airflow."
        ]
    },
    "Peach___healthy": {
        "display": "Healthy Peach",
        "severity": "None",
        "description": "Peach tree is healthy.",
        "fertilizer": "Balanced NPK and organic compost.",
        "pesticide": "No treatment needed; scout regularly.",
        "irrigation": "Deep watering; avoid wet foliage.",
        "prevention": [
            "Annual pruning for airflow.",
            "Monitor for pests and diseases.",
            "Remove dead wood and debris.",
            "Maintain proper spacing."
        ]
    },
    "Pepper,_bell___Bacterial_spot": {
        "display": "Bell Pepper Bacterial Spot",
        "severity": "Moderate",
        "description": "Small dark water-soaked lesions on leaves and fruits.",
        "fertilizer": "Balanced NPK; avoid excessive nitrogen. Add calcium to reduce fruit cracking.",
        "pesticide": "Copper-based bactericides at early symptom stages.",
        "irrigation": "Drip irrigation to avoid wetting foliage.",
        "prevention": [
            "Use certified disease-free seeds.",
            "Remove infected plants.",
            "Rotate crops.",
            "Sanitize tools and hands.",
            "Maintain proper spacing."
        ]
    },
    "Pepper,_bell___healthy": {
        "display": "Healthy Bell Pepper",
        "severity": "None",
        "description": "Plant is healthy.",
        "fertilizer": "Balanced NPK; compost incorporation.",
        "pesticide": "No chemical needed.",
        "irrigation": "Maintain consistent soil moisture; drip preferred.",
        "prevention": [
            "Monitor regularly.",
            "Prune for airflow.",
            "Remove debris.",
            "Rotate crops annually."
        ]
    },
    "Potato___Early_blight": {
        "display": "Potato Early Blight",
        "severity": "Moderate",
        "description": "Fungal leaf spots and stem lesions reducing yield.",
        "fertilizer": "Balanced NPK with potassium emphasis. Avoid high nitrogen.",
        "pesticide": "Fungicides like Mancozeb or Chlorothalonil at first signs of infection.",
        "irrigation": "Water at base; avoid wetting leaves.",
        "prevention": [
            "Rotate crops.",
            "Remove infected foliage.",
            "Use certified seed potatoes.",
            "Ensure proper spacing.",
            "Monitor regularly."
        ]
    },
    "Potato___Late_blight": {
        "display": "Potato Late Blight",
        "severity": "Severe",
        "description": "Rapidly spreading fungal disease causing dark lesions on leaves and tubers.",
        "fertilizer": "Balanced NPK with moderate nitrogen. Add compost for soil health.",
        "pesticide": "Fungicides like Mancozeb or Metalaxyl applied preventively and at first symptom.",
        "irrigation": "Avoid wetting leaves; use drip irrigation.",
        "prevention": [
            "Plant resistant varieties.",
            "Remove infected plants immediately.",
            "Rotate crops.",
            "Avoid overhead watering.",
            "Monitor frequently in humid weather."
        ]
    },
    "Potato___healthy": {
        "display": "Healthy Potato",
        "severity": "None",
        "description": "Potato plant is healthy.",
        "fertilizer": "Balanced NPK; incorporate organic matter.",
        "pesticide": "No treatment needed; preventive monitoring.",
        "irrigation": "Maintain soil moisture; drip irrigation preferred.",
        "prevention": [
            "Use certified seed potatoes.",
            "Rotate crops.",
            "Keep field weed-free.",
            "Monitor regularly."
        ]
    },
    "Raspberry___healthy": {
        "display": "Healthy Raspberry",
        "severity": "None",
        "description": "Raspberry plants are healthy.",
        "fertilizer": "Balanced NPK with compost; add magnesium and boron as needed.",
        "pesticide": "No chemical treatment required.",
        "irrigation": "Drip irrigation to maintain consistent soil moisture.",
        "prevention": [
            "Prune annually to remove dead wood.",
            "Mulch to conserve moisture.",
            "Monitor for pests.",
            "Maintain proper spacing."
        ]
    },
    "Soybean___healthy": {
        "display": "Healthy Soybean",
        "severity": "None",
        "description": "Soybean plants are healthy.",
        "fertilizer": "Apply NPK based on soil test; inoculate seeds with rhizobia if needed.",
        "pesticide": "No chemical required.",
        "irrigation": "Maintain soil moisture; avoid waterlogging.",
        "prevention": [
            "Rotate crops.",
            "Monitor for early disease symptoms.",
            "Maintain proper spacing."
        ]
    },
    "Squash___Powdery_mildew": {
        "display": "Squash Powdery Mildew",
        "severity": "Moderate",
        "description": "White powdery fungal growth on leaves and stems.",
        "fertilizer": "Balanced NPK with emphasis on potassium; avoid excess nitrogen.",
        "pesticide": "Use sulfur-based fungicides or Potassium bicarbonate sprays weekly.",
        "irrigation": "Water at soil level; avoid wetting leaves.",
        "prevention": [
            "Prune affected leaves.",
            "Provide good airflow.",
            "Rotate crops.",
            "Use resistant varieties.",
            "Monitor regularly."
        ]
    },
    "Strawberry___Leaf_scorch": {
        "display": "Strawberry Leaf Scorch",
        "severity": "Moderate",
        "description": "Fungal disease causing leaf necrosis and reduced yield.",
        "fertilizer": "Balanced NPK; add calcium and magnesium.",
        "pesticide": "Apply Captan or Thiram fungicides preventively.",
        "irrigation": "Avoid wetting foliage; water at root zone.",
        "prevention": [
            "Remove infected leaves.",
            "Prune for airflow.",
            "Use disease-free planting material.",
            "Rotate fields."
        ]
    },
    "Strawberry___healthy": {
        "display": "Healthy Strawberry",
        "severity": "None",
        "description": "Strawberry plants are healthy.",
        "fertilizer": "Balanced NPK; organic mulch.",
        "pesticide": "No chemical needed.",
        "irrigation": "Drip irrigation; maintain consistent moisture.",
        "prevention": [
            "Monitor for pests and disease.",
            "Prune dead leaves.",
            "Maintain proper spacing."
        ]
    },
    "Tomato___Bacterial_spot": {
        "display": "Tomato Bacterial Spot",
        "severity": "Moderate",
        "description": "Water-soaked spots on leaves and fruits causing defoliation.",
        "fertilizer": "Balanced NPK; add calcium to reduce fruit cracking.",
        "pesticide": "Copper-based bactericides at early infection stages.",
        "irrigation": "Drip irrigation; avoid wetting foliage.",
        "prevention": [
            "Use disease-free seeds.",
            "Remove infected plants.",
            "Rotate crops.",
            "Maintain spacing.",
            "Sanitize tools."
        ]
    },
    "Tomato___Early_blight": {
        "display": "Tomato Early Blight",
        "severity": "Moderate",
        "description": "Fungal leaf spots causing defoliation and yield loss.",
        "fertilizer": "Balanced NPK; potassium to strengthen leaves.",
        "pesticide": "Apply Mancozeb or Chlorothalonil at first signs of infection.",
        "irrigation": "Water at root; avoid wetting leaves.",
        "prevention": [
            "Rotate crops annually.",
            "Remove infected foliage.",
            "Plant resistant varieties.",
            "Monitor regularly."
        ]
    },
    "Tomato___Late_blight": {
        "display": "Tomato Late Blight",
        "severity": "Severe",
        "description": "Rapidly spreading fungal disease causing dark, water-soaked lesions on leaves, stems, and fruits, leading to severe yield loss.",
        "fertilizer": "Apply balanced NPK (10-10-10) and potassium-rich fertilizers to strengthen plant tissue. Incorporate compost to improve soil health.",
        "pesticide": "Use preventive and curative fungicides like Metalaxyl-M or Mancozeb immediately upon first symptoms. Repeat every 7–10 days in wet conditions.",
        "irrigation": "Use drip irrigation; avoid overhead sprinklers. Water early morning and avoid wetting foliage to reduce infection.",
        "prevention": [
            "Plant resistant tomato varieties.",
            "Remove and destroy infected plants promptly.",
            "Practice crop rotation to prevent pathogen buildup.",
            "Ensure adequate spacing for airflow between plants.",
            "Monitor crops frequently during wet weather."
        ]
    },
    "Tomato___Leaf_Mold": {
        "display": "Tomato Leaf Mold",
        "severity": "Moderate",
        "description": "Fungal infection causing yellow patches on upper leaf surfaces with olive-green sporulation underneath, reducing photosynthesis.",
        "fertilizer": "Apply balanced NPK with emphasis on potassium to strengthen leaves and improve disease resistance.",
        "pesticide": "Spray Chlorothalonil or Mancozeb at the first sign of infection; repeat weekly if humidity is high.",
        "irrigation": "Reduce greenhouse humidity; ensure adequate ventilation and avoid overhead irrigation.",
        "prevention": [
            "Prune lower leaves and suckers to improve airflow.",
            "Sanitize greenhouse tools and benches regularly.",
            "Remove infected leaf material immediately.",
            "Maintain proper plant spacing and ventilation.",
            "Avoid excessive nitrogen that promotes dense foliage."
        ]
    },
    "Tomato___Septoria_leaf_spot": {
        "display": "Tomato Septoria Leaf Spot",
        "severity": "Moderate",
        "description": "Small circular spots with gray centers and black pycnidia appear on leaves, causing defoliation and reduced yield.",
        "fertilizer": "Moderate nitrogen application; maintain phosphorus and potassium for overall plant health.",
        "pesticide": "Apply fungicides such as Chlorothalonil or Mancozeb weekly when conditions favor infection.",
        "irrigation": "Drip irrigation only; avoid wetting leaves. Water early morning to allow foliage to dry quickly.",
        "prevention": [
            "Remove infected leaves and debris from the field.",
            "Practice crop rotation with non-host crops.",
            "Ensure proper spacing for airflow.",
            "Stake or trellis plants to reduce leaf contact with soil.",
            "Monitor plants regularly for early spotting."
        ]
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "display": "Tomato Spider Mites",
        "severity": "Moderate",
        "description": "Bronzed or yellow leaves with fine webbing underneath, reducing photosynthetic capacity and plant vigor.",
        "fertilizer": "Avoid excessive nitrogen which encourages mite outbreaks. Maintain balanced nutrient levels with potassium and calcium.",
        "pesticide": "Spray Abamectin or neem oil. Introduce predatory mites like Phytoseiulus persimilis in greenhouses.",
        "irrigation": "Maintain consistent soil moisture and high humidity to suppress mite population. Avoid water stress.",
        "prevention": [
            "Inspect leaves weekly for early signs.",
            "Remove weeds that host mites.",
            "Rotate crops or interplant with non-host species.",
            "Use reflective mulches to deter mites.",
            "Maintain proper ventilation in greenhouse setups."
        ]
    },
    "Tomato___Target_Spot": {
        "display": "Tomato Target Spot",
        "severity": "Moderate",
        "description": "Circular lesions with concentric rings form a 'target' pattern on leaves and fruits, reducing yield quality.",
        "fertilizer": "Balanced NPK to strengthen plants; include calcium and magnesium to improve tissue resilience.",
        "pesticide": "Preventive fungicides like Azoxystrobin or Chlorothalonil can reduce spread. Apply at first symptoms and repeat as needed.",
        "irrigation": "Avoid overhead irrigation. Use drip or furrow irrigation to keep foliage dry.",
        "prevention": [
            "Stake plants to improve air circulation.",
            "Remove and destroy infected plant material.",
            "Practice crop rotation with non-host crops.",
            "Monitor environmental conditions and humidity.",
            "Sanitize tools and hands before handling plants."
        ]
    },
    "Tomato___Tomato_mosaic_virus": {
        "display": "Tomato Mosaic Virus",
        "severity": "Severe",
        "description": "Viral disease causing mosaic patterns on leaves, leaf distortion, and internal browning of fruits, transmitted mechanically or via contaminated tools.",
        "fertilizer": "Maintain balanced nutrient supply to support plant health; avoid excess nitrogen to prevent stress.",
        "pesticide": "No direct antiviral chemical; focus on vector control and hygiene.",
        "irrigation": "Maintain consistent soil moisture to reduce plant stress, but avoid waterlogging.",
        "prevention": [
            "Use certified virus-free seeds and seedlings.",
            "Remove and destroy infected plants immediately.",
            "Sanitize tools and equipment between uses.",
            "Limit mechanical transmission by workers.",
            "Control vectors if identified (thrips or sap-sucking insects)."
        ]
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "display": "Tomato Yellow Leaf Curl Virus",
        "severity": "Severe",
        "description": "Yellowing and upward leaf curling due to viral infection, primarily transmitted by whiteflies, causing reduced growth and yield.",
        "fertilizer": "Enhance nutrition with potassium and micronutrients to reduce stress and improve plant vigor.",
        "pesticide": "Control whitefly vectors using insecticidal soap, neem oil, or approved systemic insecticides.",
        "irrigation": "Provide consistent soil moisture; avoid drought stress which exacerbates virus impact.",
        "prevention": [
            "Remove and destroy infected plants.",
            "Plant resistant or tolerant tomato varieties.",
            "Implement physical barriers like insect netting in greenhouses.",
            "Monitor and control whitefly populations regularly.",
            "Maintain field hygiene and weed control."
        ]
    },
    "Tomato___healthy": {
        "display": "Healthy Tomato",
        "severity": "None",
        "description": "No disease detected; plants are thriving with optimal growth.",
        "fertilizer": "Balanced NPK with supplemental calcium and magnesium. Compost or organic matter enhances soil health.",
        "pesticide": "Preventive sprays optional based on field scouting; minimal chemical use.",
        "irrigation": "Deep watering 1–2 inches per week, preferably via drip irrigation.",
        "prevention": [
            "Monitor for early symptoms of pests and diseases.",
            "Stake or trellis plants for better airflow.",
            "Maintain proper spacing and pruning practices.",
            "Rotate crops annually.",
            "Keep records of soil tests, fertilization, and plant health."
        ]
    },
    "unknown": {
        "display": "Unclassified / Low Confidence",
        "severity": "Unknown",
        "description": "The model could not confidently identify a disease on the uploaded leaf image.",
        "fertilizer": "Apply general balanced NPK fertilizer to maintain plant health.",
        "pesticide": "Consult a local agricultural expert for targeted treatment.",
        "irrigation": "Maintain regular irrigation without waterlogging.",
        "prevention": [
            "Retake photo with good lighting and clear leaf visibility.",
            "Ensure leaf fills the frame for accurate detection.",
            "Avoid overlapping leaves or stems in the photo."
        ]
    }
}
# ──────────────────────────────────────────────────────────────────
# NLP query knowledge base
# ──────────────────────────────────────────────────────────────────
QUERY_RESPONSES = {
    "yellow_leaves": {
        "keywords": ["yellow", "yellowing", "pale", "chlorosis", "pale leaves"],
        "title": "🍃 Yellow Leaves Diagnosis",
        "response": (
            "Yellowing leaves (chlorosis) can have several causes:\n\n"
            "(1) Nitrogen deficiency — the most common cause. Apply urea (46-0-0) or ammonium nitrate at 20–25 kg/ha.\n"
            "(2) Iron or Magnesium deficiency — apply chelated micronutrient foliar spray (FeSO4 or MgSO4).\n"
            "(3) Overwatering / waterlogging — check drainage, reduce irrigation frequency.\n"
            "(4) Viral infection (TYLCV, Mosaic) — check for insect vectors like whiteflies and aphids.\n"
            "(5) Natural senescence of older/lower leaves — normal, no action required.\n\n"
            "📌 Key diagnostic tip: Uniform yellowing = nutrient issue. Irregular spots with yellow halo = disease."
        )
    },
    "brown_spots": {
        "keywords": ["brown", "spots", "lesions", "blight", "dark spots", "black spots"],
        "title": "🔴 Brown Spots & Lesions",
        "response": (
            "Brown or dark spots on crop leaves typically indicate:\n\n"
            "(1) Early Blight (Alternaria) — bull's-eye circular lesions. Apply Mancozeb or Chlorothalonil.\n"
            "(2) Late Blight (Phytophthora) — water-soaked, rapidly spreading. Apply Metalaxyl + Mancozeb URGENTLY.\n"
            "(3) Bacterial leaf spot — angular, water-soaked turning brown. Apply copper-based bactericide.\n"
            "(4) Septoria leaf spot — small circular spots with gray centers. Apply Chlorothalonil.\n"
            "(5) Sunscald — bleached/papery patches on sun-exposed side — provide afternoon shade.\n\n"
            "📌 Spot shape is diagnostic: circular = fungal; angular = bacterial; irregular = abiotic stress."
        )
    },
    "wilting": {
        "keywords": ["wilt", "wilting", "drooping", "collapsing", "droopy", "limp"],
        "title": "💧 Wilting Plant Diagnosis",
        "response": (
            "Wilting indicates water stress or vascular disease:\n\n"
            "(1) Water deficit — check soil at 15 cm depth; if dry, irrigate deeply (25–30 mm).\n"
            "(2) Fusarium wilt — stem cross-section shows brown vascular discoloration. No cure; remove plants.\n"
            "(3) Verticillium wilt — similar symptoms, cooler conditions. Remove infected plants.\n"
            "(4) Root rot (Phytophthora) — from overwatering; improve drainage, apply Metalaxyl soil drench.\n"
            "(5) Nematode damage — soil test for RKN; apply Carbofuran nematicide if confirmed.\n"
            "(6) Heat stress — install 30% shade net during peak summer temperatures.\n\n"
            "📌 If wilting only occurs midday and recovers at night, suspect water stress or heat — not disease."
        )
    },
    "insects": {
        "keywords": ["insect", "bug", "pest", "aphid", "whitefly", "mite", "caterpillar", "worm", "fly"],
        "title": "🐛 Pest Management Advisory",
        "response": (
            "Common crop pest management strategies:\n\n"
            "(1) Aphids — Imidacloprid 17.8 SL at 0.5 ml/L; neem oil 5 ml/L; release Chrysoperla predators.\n"
            "(2) Whiteflies — Yellow sticky traps (25/ha), Thiamethoxam 25 WG; silver reflective mulch.\n"
            "(3) Spider mites — Increase humidity; Abamectin 1.9 EC at 0.5 ml/L; spray leaf undersides.\n"
            "(4) Caterpillars/FAW — Bt (Bacillus thuringiensis) spray 2 g/L; Spinosad; manual removal at night.\n"
            "(5) Leaf miners — Abamectin; remove heavily infested leaves; yellow sticky traps.\n\n"
            "📌 IPM principle: Scout twice weekly. Use chemical control only when pest crosses economic threshold."
        )
    },
    "fertilizer": {
        "keywords": ["fertilizer", "nutrition", "nutrient", "feed", "compost", "npk", "deficiency", "urea", "dap"],
        "title": "🌱 Fertilizer & Nutrition",
        "response": (
            "Crop fertilization guidelines:\n\n"
            "(1) Nitrogen (N) — promotes vegetative growth. Apply urea in 3 splits: basal, 30 DAS, 60 DAS.\n"
            "(2) Phosphorus (P) — root development and flowering. Apply DAP (18-46-0) as basal dose.\n"
            "(3) Potassium (K) — fruit quality and disease resistance. Apply MOP or SOP at flowering stage.\n"
            "(4) Zinc deficiency — interveinal chlorosis on young leaves. Apply ZnSO4 at 25 kg/ha.\n"
            "(5) Iron deficiency — young leaf yellowing. Apply chelated iron (FeSO4) as foliar spray.\n"
            "(6) Organic option — FYM compost at 10 tonnes/ha improves soil structure and microbial activity.\n\n"
            "📌 Always soil-test before fertilizing. Over-fertilization increases disease susceptibility."
        )
    },
    "irrigation": {
        "keywords": ["water", "irrigation", "moisture", "drought", "rain", "flood", "waterlog", "drip"],
        "title": "💦 Irrigation Advisory",
        "response": (
            "Optimal irrigation practices for crop health:\n\n"
            "(1) Drip irrigation — 40–50% water saving vs. flood; keeps foliage dry reducing disease risk significantly.\n"
            "(2) Scheduling — irrigate when soil moisture at 60% field capacity. Use tensiometer (20–30 kPa reading).\n"
            "(3) Critical stages — never allow water stress at flowering, fruit set, and grain fill stages.\n"
            "(4) Timing — morning irrigation allows foliage to dry by evening, greatly reducing fungal disease.\n"
            "(5) Mulching — 5–8 cm straw or plastic mulch reduces evaporation by 50–60%.\n"
            "(6) Waterlogging — raise beds, install drainage channels; most vegetables die within 24–48 hours.\n\n"
            "📌 Rule of thumb: most vegetables need 25–30 mm per week. Measure actual rainfall with a rain gauge."
        )
    },
    "fungal": {
        "keywords": ["fungus", "fungal", "mold", "mildew", "rust", "rot", "blight", "Phytophthora", "Fusarium"],
        "title": "🍄 Fungal Disease Management",
        "response": (
            "Fungal disease prevention and control strategy:\n\n"
            "(1) Prevention (best) — avoid leaf wetness, improve airflow, use resistant varieties, crop rotation.\n"
            "(2) Cultural control — remove infected material immediately; never compost diseased plants.\n"
            "(3) Protectant fungicides — Mancozeb 75WP, Chlorothalonil 75WP. Apply before infection period.\n"
            "(4) Systemic fungicides — Azoxystrobin, Propiconazole, Metalaxyl. Apply at early symptoms.\n"
            "(5) Organic options — Copper hydroxide (Blitox), Trichoderma viride bio-fungicide.\n"
            "(6) Resistance management — rotate fungicide chemical groups every 2–3 applications.\n\n"
            "📌 Apply fungicides in early morning or evening. Avoid application during rain or extreme heat."
        )
    },
    "default": {
        "title": "🌾 General Agricultural Advisory",
        "response": (
            "Thank you for your query. Here is general crop management advice:\n\n"
            "(1) Integrated Pest Management (IPM) — combines cultural, biological, and chemical controls.\n"
            "(2) Regular field scouting — 2–3 times per week for early disease/pest detection.\n"
            "(3) Crop rotation — prevents buildup of soil-borne pathogens and pests.\n"
            "(4) Quality inputs — use certified seed, balanced fertilizers, and registered pesticides.\n"
            "(5) Record keeping — maintain field diary for inputs, observations, and yields.\n"
            "(6) Expert support — contact your local Krishi Vigyan Kendra (KVK) for region-specific advice.\n\n"
            "📌 Tip: Use the Image Scan feature to upload a leaf photo for AI-powered disease identification."
        )
    }
}


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def compress_image(path: str, max_px: int = 900, quality: int = 85):
    try:
        from PIL import Image
        img = Image.open(path)
        img.thumbnail((max_px, max_px), Image.LANCZOS)
        fmt = "JPEG" if path.lower().endswith((".jpg", ".jpeg")) else "PNG"
        img.save(path, fmt, quality=quality, optimize=True)
    except Exception as e:
        logger.warning(f"Image compression failed: {e}")


def preprocess_image(path: str) -> np.ndarray:
    from PIL import Image
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def predict_disease(image_path: str) -> dict:
    if not MODEL_LOADED or cnn_model is None:
        return {
            "disease_key": "unknown",
            "disease": "Model Not Loaded",
            "confidence": 0,
            "status": "model_not_loaded",
            "message": "CNN model not found — run: python train_image_model.py --demo"
        }
    try:
        img = preprocess_image(image_path)
        preds = cnn_model.predict(img, verbose=0)
        idx = int(np.argmax(preds[0]))
        conf = float(preds[0][idx]) * 100
        key = class_names[idx] if class_names and idx < len(class_names) else f"class_{idx}"
        return {"disease_key": key,
                "disease": DISEASE_INFO.get(key, {}).get("display", key.replace("_", " ")),
                "confidence": round(conf, 2), "status": "success"}
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return {"disease_key": "unknown", "disease": "Prediction Error",
                "confidence": 0, "status": "error", "message": str(e)}


def get_query_response(query: str) -> dict:
    q = query.lower()
    # Try loaded text model first
    if TEXT_MODEL_LOADED and text_model and text_vectorizer:
        try:
            vec = text_vectorizer.transform([query])
            pred = text_model.predict(vec)[0]
            cat = text_label_encoder.inverse_transform([pred])[0] if text_label_encoder else pred
            if cat in QUERY_RESPONSES:
                return QUERY_RESPONSES[cat]
        except Exception as e:
            logger.warning(f"Text model fallback: {e}")
    # Keyword fallback
    for key, data in QUERY_RESPONSES.items():
        if key == "default":
            continue
        if any(kw in q for kw in data.get("keywords", [])):
            return data
    return QUERY_RESPONSES["default"]


# ──────────────────────────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────────────────────────

def load_models():
    global cnn_model, class_names, text_model, text_vectorizer, text_label_encoder
    global MODEL_LOADED, TEXT_MODEL_LOADED

    # CNN — support both .h5 and .keras formats
    try:
        from tensorflow.keras.models import load_model as keras_load
        h5    = "models/plant_disease_model.h5"
        keras_fmt = "models/plant_disease_model.keras"
        model_path = h5 if os.path.exists(h5) else (keras_fmt if os.path.exists(keras_fmt) else None)
        if model_path:
            cnn_model = keras_load(model_path)
            MODEL_LOADED = True
            logger.info(f"✅  CNN model loaded from {model_path}")
        else:
            logger.warning("⚠️   CNN model not found. Run: python create_demo.py")
    except Exception as e:
        logger.error(f"CNN load error: {e}")

    # Class names
    try:
        cn_path = "models/class_names.json"
        if os.path.exists(cn_path):
            with open(cn_path) as f:
                class_names = json.load(f)
            logger.info(f"✅  {len(class_names)} class names loaded")
    except Exception as e:
        logger.error(f"Class names error: {e}")

    # Text model
    try:
        import pickle
        pkl = "models/text_model.pkl"
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                saved = pickle.load(f)
            text_model          = saved.get("model")
            text_vectorizer     = saved.get("vectorizer")
            text_label_encoder  = saved.get("label_encoder")
            TEXT_MODEL_LOADED   = True
            logger.info("✅  Text model loaded")
    except Exception as e:
        logger.warning(f"Text model not found: {e}")


# ──────────────────────────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    from database import get_recent_history, get_stats
    recent = get_recent_history(6)
    stats  = get_stats()
    return render_template("index.html", recent=recent, stats=stats, model_loaded=MODEL_LOADED)


@app.route("/scan", methods=["GET", "POST"])
def scan():
    if request.method == "GET":
        return render_template("upload.html", model_loaded=MODEL_LOADED)

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400
    f = request.files["image"]
    if not f.filename:
        return jsonify({"error": "No file selected"}), 400
    if not allowed_file(f.filename):
        return jsonify({"error": "Invalid format. Use JPG, PNG, or WebP"}), 400

    try:
        ext      = f.filename.rsplit(".", 1)[1].lower()
        fname    = f"{uuid.uuid4().hex}.{ext}"
        fpath    = os.path.join(app.config["UPLOAD_FOLDER"], fname)
        f.save(fpath)
        compress_image(fpath)

        result   = predict_disease(fpath)
        key      = result.get("disease_key", "unknown")
        info     = DISEASE_INFO.get(key, DISEASE_INFO["unknown"])

        from database import save_prediction
        save_prediction(image_path=fpath, disease=result["disease"],
                        confidence=result["confidence"])

        return jsonify({
            "status":      result.get("status"),
            "disease":     result["disease"],
            "disease_key": key,
            "confidence":  result["confidence"],
            "severity":    info["severity"],
            "description": info["description"],
            "treatment": {
                "fertilizer": info["fertilizer"],
                "pesticide":  info["pesticide"],
                "irrigation": info["irrigation"],
                "prevention": info["prevention"]
            },
            "image_url": f"/static/uploads/{fname}",
            "message":   result.get("message", "")
        })
    except Exception as e:
        logger.error(f"Scan error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["GET", "POST"])
def ask():
    if request.method == "GET":
        return render_template("query.html", model_loaded=MODEL_LOADED)

    data = request.get_json(silent=True) or {}
    query = data.get("query", "").strip()
    if len(query) < 3:
        return jsonify({"error": "Query too short"}), 400

    resp = get_query_response(query)

    from database import save_prediction
    save_prediction(image_path=None, disease=resp.get("title", "Query"),
                    confidence=None, query_text=query)

    return jsonify({"title": resp["title"], "response": resp["response"], "status": "success"})


@app.route("/history")
def history():
    from database import get_all_history
    records = get_all_history()
    return render_template("history.html", records=records, model_loaded=MODEL_LOADED)


@app.route("/api/history/clear", methods=["POST"])
def clear_history():
    from database import clear_all_history
    clear_all_history()
    return jsonify({"status": "cleared"})


@app.route("/api/status")
def api_status():
    return jsonify({"model_loaded": MODEL_LOADED, "text_model_loaded": TEXT_MODEL_LOADED,
                    "classes": len(class_names)})

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        user_message = data.get('message', '').strip()
        if not user_message:
            return jsonify({'response': 'Please type a message.'})

        msg = user_message.lower().strip()

        # ── KNOWLEDGE BASE ──
        DISEASE_INFO = {
            "tomato early blight": {
                "cause": "Fungal infection by Alternaria solani.",
                "symptoms": "Dark brown circular spots with concentric rings on older leaves. Yellowing around spots.",
                "treatment": "Spray Mancozeb (2g/L) or Chlorothalonil every 7-10 days. Remove infected leaves immediately.",
                "prevention": "Crop rotation, avoid overhead irrigation, use resistant varieties, maintain plant spacing."
            },
            "tomato late blight": {
                "cause": "Water mould Phytophthora infestans.",
                "symptoms": "Water-soaked greyish-green patches on leaves turning dark brown. White mould on leaf underside.",
                "treatment": "Apply Metalaxyl+Mancozeb or Cymoxanil fungicide. Destroy heavily infected plants.",
                "prevention": "Avoid overhead watering, ensure drainage, plant in well-ventilated areas."
            },
            "tomato bacterial spot": {
                "cause": "Bacteria Xanthomonas vesicatoria.",
                "symptoms": "Small water-soaked spots turning brown with yellow halo on leaves and fruits.",
                "treatment": "Spray copper-based bactericide. Remove infected material immediately.",
                "prevention": "Use disease-free seeds, avoid working in wet fields, crop rotation."
            },
            "tomato leaf mold": {
                "cause": "Fungus Passalora fulva.",
                "symptoms": "Pale green or yellow spots on upper leaf surface, olive-green mould on underside.",
                "treatment": "Apply Mancozeb or Thiram fungicide. Improve ventilation.",
                "prevention": "Reduce humidity, increase plant spacing, use resistant varieties."
            },
            "tomato mosaic virus": {
                "cause": "Tomato Mosaic Virus spread by contact and insects.",
                "symptoms": "Mosaic pattern of light and dark green on leaves, distorted leaf growth, stunted plant.",
                "treatment": "No cure. Remove and destroy infected plants immediately.",
                "prevention": "Use certified virus-free seeds, control aphid vectors."
            },
            "potato late blight": {
                "cause": "Phytophthora infestans pathogen.",
                "symptoms": "Dark water-soaked lesions on leaves and stems. White fungal growth underneath.",
                "treatment": "Apply copper-based or Metalaxyl fungicide. Remove infected plants.",
                "prevention": "Use certified disease-free seed potatoes, avoid waterlogging."
            },
            "potato early blight": {
                "cause": "Fungal infection by Alternaria solani.",
                "symptoms": "Dark brown spots with concentric rings on older leaves.",
                "treatment": "Spray Mancozeb or Chlorothalonil at 7-10 day intervals.",
                "prevention": "Avoid overhead irrigation, remove crop debris."
            },
            "corn common rust": {
                "cause": "Fungal pathogen Puccinia sorghi.",
                "symptoms": "Small oval brick-red pustules on both leaf surfaces.",
                "treatment": "Apply Propiconazole or Azoxystrobin fungicide at early stage.",
                "prevention": "Plant rust-resistant hybrids, scout fields regularly."
            },
            "corn northern leaf blight": {
                "cause": "Fungus Exserohilum turcicum.",
                "symptoms": "Long grey-green cigar-shaped lesions on leaves turning tan.",
                "treatment": "Apply Azoxystrobin or Propiconazole fungicide.",
                "prevention": "Plant resistant hybrids, crop rotation, bury residues."
            },
            "rice brown spot": {
                "cause": "Fungus Cochliobolus miyabeanus.",
                "symptoms": "Oval brown spots with grey centers on leaves.",
                "treatment": "Spray Mancozeb or Iprodione fungicide.",
                "prevention": "Use disease-free seeds, balanced fertilization."
            },
            "rice leaf blast": {
                "cause": "Fungus Magnaporthe oryzae.",
                "symptoms": "Diamond-shaped lesions with grey centres and brown borders.",
                "treatment": "Apply Tricyclazole or Isoprothiolane fungicide immediately.",
                "prevention": "Avoid excess nitrogen, use resistant varieties."
            },
            "rice neck blast": {
                "cause": "Magnaporthe oryzae infecting the neck of panicle.",
                "symptoms": "Infected neck turns grey causing panicle to break. Grain is empty.",
                "treatment": "Apply Tricyclazole at booting stage.",
                "prevention": "Avoid high nitrogen doses, use resistant varieties."
            },
            "wheat brown rust": {
                "cause": "Fungus Puccinia triticina.",
                "symptoms": "Small round orange-brown pustules on upper leaf surface.",
                "treatment": "Spray Propiconazole or Tebuconazole fungicide.",
                "prevention": "Plant resistant varieties, early sowing."
            },
            "wheat yellow rust": {
                "cause": "Fungus Puccinia striiformis.",
                "symptoms": "Yellow-orange pustules arranged in stripes along leaf veins.",
                "treatment": "Apply Propiconazole fungicide immediately.",
                "prevention": "Use resistant varieties, avoid late sowing."
            },
            "apple scab": {
                "cause": "Fungus Venturia inaequalis.",
                "symptoms": "Olive-green to black scab lesions on leaves and fruits.",
                "treatment": "Apply Captan or Myclobutanil fungicide from pink bud stage.",
                "prevention": "Remove fallen leaves, prune for airflow."
            },
            "grape black rot": {
                "cause": "Fungus Guignardia bidwellii.",
                "symptoms": "Tan circular spots on leaves. Berries shrivel into black mummies.",
                "treatment": "Apply Mancozeb or Myclobutanil from budbreak.",
                "prevention": "Remove mummified berries, prune for airflow."
            },
            "grape powdery mildew": {
                "cause": "Fungus Erysiphe necator.",
                "symptoms": "White powdery patches on leaves, shoots, and berries.",
                "treatment": "Apply Sulfur-based fungicide or Myclobutanil every 14 days.",
                "prevention": "Prune for airflow, avoid excessive nitrogen."
            },
        }

        CROP_INFO = {
            "tomato": {
                "description": "Tomato is a warm-season crop requiring full sun and well-drained fertile soil.",
                "soil": "Loamy soil with pH 6.0-6.8. Add organic compost before planting.",
                "irrigation": "Regular deep watering. Drip irrigation preferred. Avoid wetting leaves.",
                "fertilizer": "Apply NPK 19:19:19 at transplanting. Use potassium-rich fertilizer during fruiting.",
                "diseases": "Early Blight, Late Blight, Bacterial Spot, Leaf Mold, Mosaic Virus.",
                "harvest": "Fruits ready 60-85 days after transplanting when fully red."
            },
            "potato": {
                "description": "Potato is a cool-season root crop growing best at 15-20 degrees C.",
                "soil": "Loose well-drained sandy loam with pH 5.5-6.5.",
                "irrigation": "Consistent moisture needed. Avoid waterlogging.",
                "fertilizer": "High potassium requirement. Apply NPK 12:32:16 at planting.",
                "diseases": "Late Blight, Early Blight, Common Scab, Blackleg.",
                "harvest": "Ready 70-120 days when foliage turns yellow."
            },
            "corn": {
                "description": "Corn (Maize) is a warm-season cereal crop requiring full sun.",
                "soil": "Well-drained fertile loam with pH 5.8-7.0.",
                "irrigation": "Critical during tasseling and silking stage.",
                "fertilizer": "High nitrogen demand. Apply urea in splits at planting, knee-high, tasseling.",
                "diseases": "Common Rust, Northern Leaf Blight, Gray Leaf Spot, Downy Mildew.",
                "harvest": "Ready 90-120 days when husks are dry and kernels hard."
            },
            "rice": {
                "description": "Rice is a staple crop grown in flooded paddy fields in tropical climates.",
                "soil": "Clay or clay-loam soil that retains water. pH 5.5-7.0.",
                "irrigation": "Requires standing water of 5-10cm during growing period.",
                "fertilizer": "Apply nitrogen in 3 splits. Use zinc sulphate for deficiency.",
                "diseases": "Brown Spot, Leaf Blast, Neck Blast, Sheath Blight, Bacterial Blight.",
                "harvest": "Ready 105-150 days when 80-85% grains turn golden yellow."
            },
            "wheat": {
                "description": "Wheat is a rabi (winter) cereal crop grown in cool dry conditions.",
                "soil": "Well-drained loam or clay-loam with pH 6.0-7.5.",
                "irrigation": "Needs 4-6 irrigations. Critical at crown root initiation and heading.",
                "fertilizer": "Apply NPK 120:60:40 kg/ha. Nitrogen in 3 splits.",
                "diseases": "Brown Rust, Yellow Rust, Powdery Mildew, Septoria, Loose Smut.",
                "harvest": "Ready 110-145 days when grain moisture is around 14%."
            },
            "apple": {
                "description": "Apple is a temperate fruit crop requiring cold winters for bud break.",
                "soil": "Well-drained loamy soil with pH 6.0-7.0.",
                "irrigation": "Drip irrigation recommended. Critical during fruit development.",
                "fertilizer": "Apply nitrogen in early spring. Potassium and calcium during fruit fill.",
                "diseases": "Apple Scab, Black Rot, Cedar Apple Rust, Powdery Mildew, Fire Blight.",
                "harvest": "Harvest 100-200 days after full bloom depending on variety."
            },
            "grape": {
                "description": "Grapes are perennial vines grown for fruit, wine, and raisins.",
                "soil": "Well-drained gravelly or sandy loam with pH 5.5-7.0.",
                "irrigation": "Moderate watering. Reduce during ripening to improve sugar.",
                "fertilizer": "Apply potassium-rich fertilizer. Avoid excess nitrogen.",
                "diseases": "Black Rot, Powdery Mildew, Downy Mildew, Botrytis, Leafroll Virus.",
                "harvest": "Harvest when berries reach target sugar level (brix)."
            },
            "sugarcane": {
                "description": "Sugarcane is a tropical grass crop grown for sugar and ethanol.",
                "soil": "Deep well-drained loam with pH 6.0-7.5.",
                "irrigation": "High water requirement. Needs 1500-2500mm water per crop.",
                "fertilizer": "High nitrogen and potassium. Apply in 3-4 splits.",
                "diseases": "Red Rot, Smut, Wilt, Grassy Shoot Disease, Ratoon Stunting.",
                "harvest": "Ready 10-18 months when sucrose content is maximum."
            },
            "cotton": {
                "description": "Cotton is a warm-season fibre crop requiring long frost-free seasons.",
                "soil": "Deep black or loamy soil with pH 6.0-8.0.",
                "irrigation": "Needs irrigation at square formation, flowering, and boll development.",
                "fertilizer": "Apply NPK 120:60:60 kg/ha. Micronutrients like boron important.",
                "diseases": "Cotton Wilt, Root Rot, Boll Rot, Leaf Spot, CLCuV Virus.",
                "harvest": "Harvest when 60% of bolls open in dry weather."
            },
            "onion": {
                "description": "Onion is a bulb crop grown in cool weather. Requires well-drained soil.",
                "soil": "Sandy loam or loamy soil with pH 6.0-7.0.",
                "irrigation": "Frequent light irrigation. Stop 2 weeks before harvest.",
                "fertilizer": "Apply NPK 100:50:50 kg/ha. Sulfur is important for onion quality.",
                "diseases": "Purple Blotch, Stemphylium Blight, Downy Mildew, Basal Rot.",
                "harvest": "Ready 100-120 days when tops fall over naturally."
            },
            "chilli": {
                "description": "Chilli is a warm-season vegetable crop with high market value.",
                "soil": "Well-drained sandy loam with pH 6.0-7.0.",
                "irrigation": "Regular irrigation needed. Avoid water stress at flowering.",
                "fertilizer": "Apply NPK 120:60:60 kg/ha with micronutrients.",
                "diseases": "Anthracnose, Bacterial Wilt, Powdery Mildew, Leaf Curl Virus.",
                "harvest": "Harvest green at 75-80 days or red at 90-100 days."
            },
        }

        PEST_INFO = {
            "aphid": "Aphids are soft-bodied sap-sucking insects. Control: Spray Imidacloprid 0.3ml/L or Dimethoate 2ml/L. Neem oil 5ml/L for organic control. Natural predators like ladybugs help significantly.",
            "whitefly": "Whiteflies cause yellowing and transmit viruses. Control: Spray Thiamethoxam 0.2g/L or use yellow sticky traps. Remove heavily infested leaves. Avoid overhead irrigation.",
            "bollworm": "Bollworms damage cotton bolls and tomato fruits. Control: Use Bt spray (Bacillus thuringiensis) or Spinosad. Install pheromone traps. Spray Chlorpyriphos at egg hatching.",
            "stem borer": "Stem borers bore into stems causing deadheart and whitear. Control: Apply Carbofuran granules 3kg/ha at base or spray Chlorpyriphos 2ml/L. Destroy infested stems.",
            "leafhopper": "Leafhoppers suck sap and transmit phytoplasmas. Control: Spray Imidacloprid or Thiamethoxam. Use reflective mulches to repel. Remove weeds around field.",
            "thrips": "Thrips cause silvering of leaves and spread viruses. Control: Spray Spinosad or Fipronil. Use blue sticky traps. Avoid dusty conditions.",
            "mite": "Spider mites cause stippling and bronzing in hot dry conditions. Control: Spray Abamectin 0.5ml/L or wettable sulfur. Increase humidity. Avoid excess nitrogen.",
            "fruit fly": "Fruit flies lay eggs in fruits causing rotting. Control: Use protein bait traps, bag developing fruits, apply Malathion bait spray. Collect and destroy fallen fruits.",
            "nematode": "Root knot nematodes cause galls on roots. Control: Apply Carbofuran 3G in soil. Use marigold as trap crop. Solarize soil before planting.",
            "grasshopper": "Grasshoppers chew leaves and stems. Control: Spray Malathion or Chlorpyriphos. Use biopesticide Metarhizium. Early morning spray is most effective.",
        }

        FERTILIZER_INFO = {
            "nitrogen": "Nitrogen promotes leafy green growth. Sources: Urea (46%N), Ammonium sulphate (21%N). Apply in splits. Deficiency causes yellowing of older leaves from bottom up.",
            "phosphorus": "Phosphorus promotes root development and flowering. Sources: DAP (18-46-0), SSP. Apply as basal dose. Deficiency causes purpling of leaves and poor root growth.",
            "potassium": "Potassium improves fruit quality and disease resistance. Sources: MOP (60% K2O), SOP. Apply at flowering. Deficiency causes leaf edge burning (scorching).",
            "npk": "Balanced NPK provides all three macronutrients. Common grades: 19:19:19 for general use, 12:32:16 for root crops, 0:52:34 for fruiting stage.",
            "urea": "Urea (46%N) is the most concentrated nitrogen fertilizer. Apply in 2-3 splits. Avoid before rain. Most commonly used for cereals and vegetables.",
            "dap": "DAP (Di-ammonium Phosphate 18:46:0) is an excellent starter fertilizer. Apply as basal dose at planting. Used for most crops.",
            "organic": "Organic fertilizers include FYM, vermicompost, and green manure. Apply 10-15 tonnes FYM per hectare. Improves soil structure and microbial activity long-term.",
            "zinc": "Zinc deficiency is common in rice and wheat. Apply Zinc Sulphate 25kg/ha as basal dose. Foliar spray of 0.5% ZnSO4 gives quick results.",
            "boron": "Boron is essential for fruit set and seed formation. Apply Borax 10kg/ha. Foliar spray 0.2% boric acid at flowering stage.",
        }

        GENERAL_QA = {
            "increase yield": "To increase crop yield: Use certified high-yielding varieties, optimize NPK based on soil test, ensure timely irrigation, control pests and diseases early, follow recommended spacing, use integrated crop management.",
            "organic farming": "Organic farming avoids synthetic chemicals. Use FYM, vermicompost, neem-based pesticides, biological control agents, crop rotation, and green manures. Get certification from recognized bodies.",
            "soil test": "Collect soil samples from 0-15cm depth from 10-15 spots per field. Send to agricultural lab. Test every 2-3 years. Helps determine nutrient status and pH for precise fertilization.",
            "crop rotation": "Crop rotation prevents disease buildup and improves fertility. Rotate cereals with legumes. Never grow the same crop family consecutively. Example: Rice-Wheat, Tomato-Maize rotation.",
            "integrated pest management": "IPM combines cultural, biological, and chemical methods. Monitor pest populations, use resistant varieties, encourage natural predators, apply pesticides only when threshold exceeded.",
            "seed treatment": "Treat seeds before sowing with Thiram or Carbendazim at 2-3g/kg seed. Biological treatment with Trichoderma 4g/kg is also effective. Improves germination and protects from soil diseases.",
            "prevent disease": "Prevention tips: Use certified seeds, treat before sowing, maintain proper spacing, avoid overhead irrigation, remove crop debris after harvest, practice crop rotation, apply preventive fungicide.",
            "greenhouse": "Greenhouse farming controls climate for high-value crops. Maintain 60-70% humidity, use drip irrigation, ensure proper ventilation to prevent fungal diseases. Suitable for vegetables and flowers.",
            "compost": "Compost improves soil organic matter and microbial activity. Mix crop residues, animal waste, and kitchen waste. Turn pile every 15 days. Ready in 45-60 days. Apply 5-10t/ha.",
            "drip irrigation": "Drip irrigation delivers water to roots saving 30-50% water. Prevents leaf wetness reducing fungal diseases. Suitable for vegetables, fruits, orchards. Install fertigation system for efficiency.",
        }

        # greeting
        if any(g in msg for g in ['hi', 'hello', 'hey', 'namaste', 'good morning', 'good evening', 'howdy']):
            return jsonify({'response': "Hello! 👋 I am your Smart Crop Advisory Assistant.\n\nAsk me about:\n• Crop diseases and treatments\n• Specific crops like tomato, rice, wheat, cotton\n• Fertilizers and nutrients\n• Irrigation methods\n• Pest management\n• General farming advice\n\nHow can I help you today?"})

        # thanks
        if any(t in msg for t in ['thank', 'thanks', 'great', 'good job', 'well done', 'helpful']):
            return jsonify({'response': "You are welcome! 🌿 Feel free to ask anytime. Happy farming!"})

        # what can you do
        if any(w in msg for w in ['what can you', 'help me', 'what do you', 'capabilities', 'features']):
            return jsonify({'response': "I can help you with:\n\n🦠 Crop Diseases - Cause, symptoms, treatment, prevention\n🌱 Crop Information - Tomato, Rice, Wheat, Potato, Corn, Cotton and more\n🐛 Pest Control - Aphids, whitefly, stem borer, bollworm and more\n🧪 Fertilizers - NPK, Urea, DAP, organic manures\n💧 Irrigation - Drip, sprinkler, flood, scheduling\n🌾 General Tips - Yield improvement, soil testing, crop rotation\n\nJust type your question!"})

        # disease query
        for disease_key, info in DISEASE_INFO.items():
            keywords = disease_key.split()
            matches = sum(1 for k in keywords if k in msg)
            if matches >= max(1, len(keywords) - 1):
                response = (
                    f"🦠 Disease: {disease_key.title()}\n\n"
                    f"📌 Cause: {info['cause']}\n\n"
                    f"🔍 Symptoms: {info['symptoms']}\n\n"
                    f"💊 Treatment: {info['treatment']}\n\n"
                    f"🛡️ Prevention: {info['prevention']}"
                )
                return jsonify({'response': response})

        # crop query
        for crop, info in CROP_INFO.items():
            if crop in msg:
                if any(w in msg for w in ['disease', 'problem', 'blight', 'rust', 'spot', 'infection', 'sick']):
                    return jsonify({'response': f"🌱 {crop.title()} Diseases:\n\n{info['diseases']}\n\nAsk me about any specific disease for detailed treatment!"})
                elif any(w in msg for w in ['fertilizer', 'fertiliser', 'nutrient', 'npk', 'urea', 'manure', 'feed']):
                    return jsonify({'response': f"🧪 {crop.title()} Fertilizer:\n\n{info['fertilizer']}"})
                elif any(w in msg for w in ['water', 'irrigat', 'drip', 'moisture']):
                    return jsonify({'response': f"💧 {crop.title()} Irrigation:\n\n{info['irrigation']}"})
                elif any(w in msg for w in ['harvest', 'pick', 'cut', 'collect', 'ready']):
                    return jsonify({'response': f"🌾 {crop.title()} Harvest:\n\n{info['harvest']}"})
                elif any(w in msg for w in ['soil', 'land', 'ground', 'ph']):
                    return jsonify({'response': f"🟫 {crop.title()} Soil:\n\n{info['soil']}"})
                elif any(w in msg for w in ['grow', 'cultivat', 'plant', 'farm', 'tell', 'about', 'info']):
                    return jsonify({'response': (
                        f"🌱 {crop.title()} Complete Guide:\n\n"
                        f"📋 About: {info['description']}\n\n"
                        f"🟫 Soil: {info['soil']}\n\n"
                        f"💧 Irrigation: {info['irrigation']}\n\n"
                        f"🧪 Fertilizer: {info['fertilizer']}\n\n"
                        f"🦠 Common Diseases: {info['diseases']}\n\n"
                        f"🌾 Harvest: {info['harvest']}"
                    )})
                else:
                    return jsonify({'response': (
                        f"🌱 {crop.title()} Overview:\n\n"
                        f"{info['description']}\n\n"
                        f"🦠 Diseases: {info['diseases']}\n\n"
                        f"🧪 Fertilizer: {info['fertilizer']}\n\n"
                        f"🌾 Harvest: {info['harvest']}\n\n"
                        f"Ask me about soil, irrigation, or specific diseases!"
                    )})

        # pest query
        for pest, info in PEST_INFO.items():
            if pest in msg:
                return jsonify({'response': f"🐛 {pest.title()} Control:\n\n{info}"})

        # fertilizer query
        for fert, info in FERTILIZER_INFO.items():
            if fert in msg:
                return jsonify({'response': f"🧪 {fert.title()}:\n\n{info}"})

        # general farming
        for topic, info in GENERAL_QA.items():
            topic_words = topic.split()
            if sum(1 for w in topic_words if w in msg) >= max(1, len(topic_words) - 1):
                return jsonify({'response': f"🌾 {topic.title()}:\n\n{info}"})

        # irrigation general
        if any(w in msg for w in ['irrigat', 'drip', 'sprinkler', 'flood irrigation', 'water stress']):
            return jsonify({'response': "💧 Irrigation Methods:\n\n• Drip Irrigation: Saves 30-50% water, best for vegetables and fruits\n• Sprinkler: Simulates rainfall, good for field crops\n• Flood: Traditional method, essential for rice\n\nAsk about a specific method for more details!"})

        # fallback using existing NLP model
        try:
            prediction = text_model.predict([msg])[0]
            label = text_labels.get(str(prediction), str(prediction))
            return jsonify({'response': f"🌿 Advisory on {label.title()}:\n\nFor {label.lower()}, follow these best practices:\n• Use certified quality seeds\n• Maintain optimal soil health\n• Apply balanced fertilization based on crop stage\n• Monitor for pests and diseases weekly\n• Follow recommended irrigation schedule\n• Practice crop rotation\n\nAsk me specifically about a crop or disease for precise guidance!"})
        except Exception:
            return jsonify({'response': "🌿 I can help with crop diseases, treatments, fertilizers, irrigation, and pest management.\n\nTry asking:\n• 'Tell me about tomato early blight'\n• 'How to grow rice?'\n• 'How to control aphids?'\n• 'What fertilizer for wheat?'"})

    except Exception as e:
        return jsonify({'response': "🌿 Please ask me about crop diseases, treatments, or farming advice!"})




# ──────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("static/uploads", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    from database import init_db
    init_db()
    load_models()

    print("\n" + "═" * 58)
    print("  🌿  FARMER ADVISORY SYSTEM  —  AI Crop Disease Detector")
    print("═" * 58)
    print(f"  🌐  http://127.0.0.1:5000")
    print(f"  🤖  CNN model  : {'✅ Ready' if MODEL_LOADED else '⚠️  Not loaded (run train_image_model.py --demo)'}")
    print(f"  💬  NLP model  : {'✅ Ready' if TEXT_MODEL_LOADED else '⚠️  Not loaded (run train_text_model.py)'}")
    print("═" * 58 + "\n")

    app.run(debug=False, host="0.0.0.0", port=10000)
    

from flask import Flask, render_template, request, jsonify, session
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import pickle, json, os, re
from sklearn.feature_extraction.text import TfidfVectorizer

# ── load your existing models (skip if already loaded) ──
cnn_model = load_model('models/plant_disease_model.h5')
with open('models/text_model.pkl', 'rb') as f:
    text_model = pickle.load(f)
with open('models/text_labels.json', 'r') as f:
    text_labels = json.load(f)
with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# ── disease advice dictionary ──
DISEASE_ADVICE = {
    "Tomato_Early_blight": {
        "cause": "Fungal infection caused by Alternaria solani.",
        "symptoms": "Dark brown spots with concentric rings on lower leaves.",
        "treatment": "Apply Mancozeb or Chlorothalonil fungicide. Remove infected leaves immediately.",
        "prevention": "Rotate crops every season. Avoid overhead watering."
    },
    "Tomato_Late_blight": {
        "cause": "Caused by Phytophthora infestans.",
        "symptoms": "Water-soaked grey-green spots turning brown on leaves and stems.",
        "treatment": "Use Metalaxyl or Cymoxanil fungicide. Destroy infected plants.",
        "prevention": "Plant resistant varieties. Ensure proper spacing for air circulation."
    },
    "Potato_Late_blight": {
        "cause": "Phytophthora infestans pathogen.",
        "symptoms": "Dark water-soaked lesions on leaves, white fungal growth underneath.",
        "treatment": "Apply copper-based fungicides. Remove and destroy all infected material.",
        "prevention": "Use certified disease-free seed potatoes. Avoid waterlogging."
    },
    "Corn_Common_rust": {
        "cause": "Fungal pathogen Puccinia sorghi.",
        "symptoms": "Small, oval, brick-red pustules scattered on both leaf surfaces.",
        "treatment": "Apply Propiconazole or Azoxystrobin fungicide at early infection stage.",
        "prevention": "Plant rust-resistant hybrids. Scout fields regularly."
    },
    "Healthy": {
        "cause": "No disease detected.",
        "symptoms": "Leaf appears healthy with no visible symptoms.",
        "treatment": "No treatment required.",
        "prevention": "Continue current crop management practices."
    }
}

# ── crop information dictionary ──
CROP_INFO = {
    "tomato": "Tomato is a warm-season crop. It requires well-drained soil, full sunlight, and regular watering. Common diseases include Early Blight, Late Blight, and Bacterial Spot.",
    "potato": "Potato grows best in cool climates with loose, well-drained soil. Major diseases are Late Blight and Early Blight. Requires consistent moisture.",
    "corn": "Corn (Maize) needs full sun and fertile soil. Common diseases include Common Rust, Northern Leaf Blight, and Gray Leaf Spot.",
    "wheat": "Wheat is a rabi crop grown in cool, dry climates. Major diseases are Brown Rust, Yellow Rust, and Septoria Leaf Blotch.",
    "rice": "Rice requires standing water and warm temperatures. Common diseases include Brown Spot, Leaf Blast, and Neck Blast.",
    "apple": "Apple trees grow in temperate climates. Major diseases include Apple Scab, Black Rot, and Cedar Apple Rust.",
    "grape": "Grapes need well-drained soil and full sun. Common diseases are Black Rot, Downy Mildew, and Powdery Mildew.",
}

# ── greeting & fallback responses ──
GREETINGS = ["hi", "hello", "hey", "good morning", "good evening", "namaste"]
GREETING_RESPONSE = "Hello! I am your Smart Crop Advisory Assistant. You can ask me about:\n• Crop diseases and their treatments\n• Crop types and growing conditions\n• Fertilizer and irrigation advice\n• Pest management tips\nHow can I help you today?"

THANKS = ["thank", "thanks", "thank you", "ok", "okay", "great", "nice"]
THANKS_RESPONSE = "You are welcome! Feel free to ask me anything about crop diseases or farming advice anytime."

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens]
    return ' '.join(tokens)

def get_disease_response(disease_name):
    # fuzzy match disease name to DISEASE_ADVICE keys
    for key in DISEASE_ADVICE:
        if key.lower().replace('_', ' ') in disease_name.lower():
            info = DISEASE_ADVICE[key]
            return (
                f"**Disease:** {disease_name.replace('_', ' ')}\n\n"
                f"**Cause:** {info['cause']}\n\n"
                f"**Symptoms:** {info['symptoms']}\n\n"
                f"**Treatment:** {info['treatment']}\n\n"
                f"**Prevention:** {info['prevention']}"
            )
    return f"Disease detected: {disease_name.replace('_', ' ')}. Please consult your local agricultural officer for specific treatment advice."

def get_crop_response(message):
    for crop, info in CROP_INFO.items():
        if crop in message.lower():
            return info
    return None

def chatbot_response(user_message):
    msg_lower = user_message.lower().strip()

    # greeting check
    if any(greet in msg_lower for greet in GREETINGS):
        return GREETING_RESPONSE

    # thanks check
    if any(t in msg_lower for t in THANKS):
        return THANKS_RESPONSE

    # crop specific query
    crop_resp = get_crop_response(msg_lower)
    if crop_resp:
        return crop_resp

    # disease name directly mentioned
    for key in DISEASE_ADVICE:
        if key.lower().replace('_', ' ') in msg_lower:
            return get_disease_response(key)

    # use your existing NLP model
    processed = preprocess_text(user_message)
    try:
        prediction = text_model.predict([processed])[0]
        label = text_labels.get(str(prediction), prediction)
        return f"Based on your query, here is my advice on **{label}**:\nFor detailed guidance on {label.lower()}, please ensure you are following recommended agricultural practices. Use certified seeds, maintain proper irrigation, and apply fertilizers as per soil test recommendations. If you have a specific crop or disease in mind, please mention it for more precise advice."
    except Exception:
        return "I am sorry, I could not understand your query. Please try asking about a specific crop or disease, for example: 'How to treat tomato blight?' or 'Tell me about rice diseases.'"
