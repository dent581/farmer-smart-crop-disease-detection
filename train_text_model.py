"""
train_text_model.py  —  Train TF-IDF + RandomForest NLP Advisory Model
=======================================================================
Usage:  python train_text_model.py
Output: models/text_model.pkl
"""

import os, pickle, json, logging
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

TRAINING_DATA = [
    # yellow_leaves
    ("Why are my tomato leaves turning yellow?",        "yellow_leaves"),
    ("My plant leaves are pale and yellow",             "yellow_leaves"),
    ("Yellowing of crop leaves in the field",           "yellow_leaves"),
    ("What causes chlorosis in vegetables?",            "yellow_leaves"),
    ("Leaves turning pale yellow from edges",           "yellow_leaves"),
    ("Yellow patches on corn leaves",                   "yellow_leaves"),
    ("Potato leaves yellowing from the bottom",         "yellow_leaves"),
    ("Wheat leaves look pale yellow and limp",          "yellow_leaves"),
    ("Why do mango leaves turn yellow in summer?",      "yellow_leaves"),
    ("Yellow leaves with green veins on tomato",        "yellow_leaves"),
    ("Upper new leaves yellowing suddenly",             "yellow_leaves"),
    ("Whole plant becoming pale yellow",                "yellow_leaves"),

    # brown_spots
    ("Brown spots appearing on tomato leaves",          "brown_spots"),
    ("Dark lesions with rings on potato leaves",        "brown_spots"),
    ("Black spots on pepper plant leaves",              "brown_spots"),
    ("Circular brown marks spreading on corn",          "brown_spots"),
    ("Leaf blight symptoms in my paddy field",          "brown_spots"),
    ("Potato plant has brown spots with rings",         "brown_spots"),
    ("Burned looking brown areas on plant leaves",      "brown_spots"),
    ("Rust colored spots on wheat crop",                "brown_spots"),
    ("Necrotic spots spreading across rice leaves",     "brown_spots"),
    ("Water-soaked spots turning brown and dry",        "brown_spots"),
    ("Small dark spots spreading fast after rain",      "brown_spots"),
    ("Angular brown lesions on bean leaves",            "brown_spots"),

    # wilting
    ("My tomato plants are wilting in afternoon",       "wilting"),
    ("Drooping leaves even after watering",             "wilting"),
    ("Tomato plants suddenly collapsing at stem base",  "wilting"),
    ("Plants wilting for no apparent reason",           "wilting"),
    ("Leaves curling and drooping in morning",          "wilting"),
    ("Stem near soil looks dark and soft",              "wilting"),
    ("Pepper plants wilting from base upward",          "wilting"),
    ("Beans wilting despite regular irrigation",        "wilting"),
    ("What is Fusarium wilt symptoms?",                 "wilting"),
    ("Verticillium wilt in potato plants",              "wilting"),
    ("Vascular wilting in tomatoes after rain",         "wilting"),
    ("Root rot causing plant to wilt",                  "wilting"),

    # insects
    ("Aphids infesting my tomato plants",               "insects"),
    ("Whiteflies on the underside of my crop leaves",   "insects"),
    ("Spider mite damage on cucumber plants",           "insects"),
    ("Caterpillars eating my vegetable plants",         "insects"),
    ("Small insects crawling on undersides of leaves",  "insects"),
    ("Leaf miner tunnels visible in leaves",            "insects"),
    ("Thrips damage on onion crop",                     "insects"),
    ("Stem borers in corn plant",                       "insects"),
    ("How to control pests organically?",               "insects"),
    ("Best insecticide for tomato whitefly",            "insects"),
    ("Sucking pests destroying my cotton crop",         "insects"),
    ("Fall armyworm in maize field",                    "insects"),

    # fertilizer
    ("What NPK fertilizer should I apply for tomatoes?","fertilizer"),
    ("How much urea should I apply per acre for wheat?","fertilizer"),
    ("Organic fertilizer recommendation for vegetables","fertilizer"),
    ("Compost application rates for small farm",        "fertilizer"),
    ("Micronutrient deficiency treatment in crops",     "fertilizer"),
    ("Zinc deficiency treatment in paddy rice",         "fertilizer"),
    ("When should I apply top dressing nitrogen?",      "fertilizer"),
    ("Fertilizer schedule for potato crop",             "fertilizer"),
    ("Potassium fertilizer for improving fruit quality","fertilizer"),
    ("Phosphorus for better root development",          "fertilizer"),
    ("Iron deficiency chlorosis treatment",             "fertilizer"),
    ("Boron deficiency in cauliflower",                 "fertilizer"),

    # irrigation
    ("How often should I water my tomato plants?",      "irrigation"),
    ("Drip irrigation setup for vegetable garden",      "irrigation"),
    ("My field is waterlogged after heavy rain",        "irrigation"),
    ("How to manage drought stress in crops",           "irrigation"),
    ("Irrigation schedule during hot summer months",    "irrigation"),
    ("Water requirement per week for wheat crop",       "irrigation"),
    ("Problems with flood irrigation system",           "irrigation"),
    ("How to conserve water in dry farming season?",    "irrigation"),
    ("Sprinkler vs drip irrigation for vegetables",     "irrigation"),
    ("Signs of overwatering in vegetable plants",       "irrigation"),
    ("Mulching for moisture retention in soil",         "irrigation"),
    ("Critical irrigation stages for maize crop",       "irrigation"),

    # fungal
    ("Fungal infection spreading across my crop field", "fungal"),
    ("White powdery coating on leaves powdery mildew",  "fungal"),
    ("Gray mold on stored tomatoes",                    "fungal"),
    ("Downy mildew control on grapes",                  "fungal"),
    ("Yellow rust disease on wheat leaves",             "fungal"),
    ("Late blight management for potato",               "fungal"),
    ("Best fungicide rotation for blight",              "fungal"),
    ("How to prevent mold spreading in field?",         "fungal"),
    ("Anthracnose disease on mango fruits",             "fungal"),
    ("Root rot caused by soil fungi",                   "fungal"),
    ("Alternaria leaf spot treatment for onion",        "fungal"),
    ("Damping off disease in seedlings nursery",        "fungal"),

    # default
    ("How do I improve overall crop yield?",            "default"),
    ("Soil health improvement techniques",              "default"),
    ("Best crops to grow in winter season",             "default"),
    ("Integrated pest management basics for farmers",   "default"),
    ("Contact agricultural extension officer KVK",      "default"),
    ("Government subsidy scheme for farmers 2024",      "default"),
    ("Crop insurance information and registration",     "default"),
    ("Organic farming certification process",           "default"),
    ("Intercropping benefits for small farmers",        "default"),
    ("Seed treatment before sowing",                    "default"),
]


def train():
    texts  = [t for t, _ in TRAINING_DATA]
    labels = [l for _, l in TRAINING_DATA]

    le = LabelEncoder()
    y  = le.fit_transform(labels)

    vec = TfidfVectorizer(ngram_range=(1, 2), max_features=2500,
                          sublinear_tf=True, strip_accents="unicode", min_df=1)
    X   = vec.fit_transform(texts)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    clf = RandomForestClassifier(n_estimators=300, max_depth=None,
                                  random_state=42, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    y_pred = clf.predict(X_te)
    log.info("\n" + classification_report(y_te, y_pred, target_names=le.classes_))
    log.info(f"Test accuracy: {(y_pred == y_te).mean():.2%}")

    os.makedirs("models", exist_ok=True)
    with open("models/text_model.pkl", "wb") as f:
        pickle.dump({"model": clf, "vectorizer": vec, "label_encoder": le}, f)
    with open("models/text_labels.json", "w") as f:
        json.dump(list(le.classes_), f, indent=2)

    log.info("✅  models/text_model.pkl saved")
    log.info("✅  models/text_labels.json saved")


if __name__ == "__main__":
    log.info("Training NLP advisory text model …")
    train()
    print("\n🚀  Done! Now run:  python app.py\n")
