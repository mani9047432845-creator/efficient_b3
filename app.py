from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os, io, requests, tempfile
import numpy as np
from PIL import Image
import torch
from torchvision import models, transforms

# =====================
# CONFIG
# =====================
app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

CLASS_LABELS = ["benign", "malignant", "normal"]
ALLOWED_EXT = {"jpg", "jpeg", "png"}
# Use GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================
# HUGGING FACE FILES
# =====================
# Path and filename updated based on your screenshot
MODEL_URL = "https://huggingface.co/mani880740255/skin_care_tflite/resolve/main/efficientnet_b3_skin_cancer.pth"

# =====================
# CHATBOT DATA
# =====================
CHAT_RESPONSES = {
    "what is skin care?": "Skin care is the practice of maintaining healthy, clean, and protected skin through proper hygiene and protection.",
    "why is skin care important?": "Proper skin care helps prevent infections, premature aging, and various skin diseases.",
    "what is a benign lesion?": "A benign skin lesion is a non-cancerous growth that does not spread to other parts of the body.",
    "what is a malignant lesion?": "A malignant skin lesion is a cancerous growth that can spread and damage surrounding tissues.",
    "difference: benign vs malignant": "Benign lesions are non-cancerous and generally harmless, while malignant lesions are cancerous and dangerous.",
    "signs of skin cancer": "Common signs include irregular shapes, color changes, bleeding, and rapid growth of a mole or spot.",
    "can benign turn malignant?": "While most benign lesions stay that way, some can become malignant if not monitored or treated properly.",
    "what causes skin cancer?": "Skin cancer is mainly caused by prolonged exposure to ultraviolet (UV) radiation from the sun or tanning beds.",
    "how to prevent skin cancer?": "Prevention involves using sunscreen (SPF 30+), wearing protective clothing, and avoiding excessive sun exposure.",
    "why is early detection key?": "Early detection significantly increases treatment success rates and reduces the risk of the cancer spreading to other organs."
}

# =====================
# HELPERS
# =====================
def allowed_file(name):
    return "." in name and name.rsplit(".", 1)[1].lower() in ALLOWED_EXT

def download_file(url):
    r = requests.get(url, stream=True)
    if r.status_code != 200:
        raise Exception(f"Model download failed: {url}")
    return io.BytesIO(r.content)

# =====================
# MODEL PREDICTION (EfficientNet-B3)
# =====================
def predict_b3(img_path):
    # 1. Download model from Hugging Face
    model_bytes = download_file(MODEL_URL)
    
    # 2. Initialize EfficientNet-B3 Architecture
    # Ensure architecture matches your training (3 classes)
    model = models.efficientnet_b3(weights=None)
    model.classifier[1] = torch.nn.Linear(1536, 3) 
    
    # 3. Load State Dict from temporary file
    with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
        tmp.write(model_bytes.read())
        tmp_path = tmp.name
    
    try:
        # Load weights and move to device
        state_dict = torch.load(tmp_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        # 4. Image Preprocessing for B3 (Standard Resize is 300x300)
        transform = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(),
            # Normalization typically used for torchvision models
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        img = Image.open(img_path).convert("RGB")
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # 5. Inference
        with torch.no_grad():
            output = model(img_tensor)
            probabilities = torch.softmax(output, dim=1)[0]
            
        idx = int(torch.argmax(probabilities))
        conf = float(probabilities[idx])
        all_probs = probabilities.cpu().tolist()
        
        return CLASS_LABELS[idx], conf, all_probs

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

# =====================
# ROUTES
# =====================
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "image required"}), 400
    
    file = request.files["image"]
    if not allowed_file(file.filename):
        return jsonify({"error": "invalid file type"}), 400
    
    path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(path)
    
    try:
        # Using the EfficientNet-B3 logic
        pred, conf, probs = predict_b3(path)
        
        if os.path.exists(path): 
            os.remove(path)
            
        return jsonify({
            "success": True,
            "prediction": pred, 
            "confidence": conf,
            "probabilities": {CLASS_LABELS[i]: float(probs[i]) for i in range(len(CLASS_LABELS))}
        })
    except Exception as e:
        if os.path.exists(path): 
            os.remove(path)
        return jsonify({"error": str(e)}), 500

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    user_msg = data.get("message", "").lower().strip()
    
    response = CHAT_RESPONSES.get(user_msg)
    
    if response:
        return jsonify({
            "reply": response, 
            "suggestions": list(CHAT_RESPONSES.keys())
        })
    
    return jsonify({
        "reply": "I am your AI Skin Assistant. Please select a question below.", 
        "suggestions": list(CHAT_RESPONSES.keys())
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)