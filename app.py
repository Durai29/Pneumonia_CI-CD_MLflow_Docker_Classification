from flask import Flask, request, render_template_string
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
import io
import datetime
import random 

# --- Configuration Constant ---
MIN_PNEUMONIA_CONFIDENCE = 0.95 # CHANGED: 95% threshold required for Pneumonia prediction

# --- Model Setup ---
CLASS_NAMES = ["Normal", "Pneumonia"]

# Load pretrained ResNet18 and adapt for binary classification
try:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    # Attempt to load the trained weights
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    print("PyTorch model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}. Using Random Dummy Model for demonstration.")
    # Fallback: Improved Dummy Model for dynamic testing
    class DummyModel:
        def __call__(self, x):
            # Simulate various prediction scenarios including low confidence
            scenario = random.randint(0, 3)
            if scenario == 0:
                # Scenario 0: High Confidence Pneumonia (e.g., 99%) - Should pass 95% threshold
                return torch.tensor([[1.0, 50.0]]) 
            elif scenario == 1:
                # Scenario 1: Medium Confidence Pneumonia (e.g., 90%) - Should FAIL 95% threshold
                return torch.tensor([[1.0, 9.0]]) 
            elif scenario == 2:
                # Scenario 2: High Confidence Normal (Always passes)
                return torch.tensor([[50.0, 1.0]])
            else:
                # Scenario 3: Medium Confidence Normal (Always passes)
                return torch.tensor([[9.0, 1.0]])
        def eval(self):
            pass
    model = DummyModel()

# --- Image Preprocessing ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


# --- Flask App ---
app = Flask(__name__)

# --- STYLISH CYBERPUNK/NEON HTML TEMPLATE (Unchanged) ---
form_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>X-RAY CLASSIFICATION [CYBERPUNK]</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Roboto+Mono:wght@400;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --neon-blue: #00ffff;
            --neon-red: #ff3366;
            --bg-dark: #0a0c1c;
            --card-dark: #12162d;
        }
        body { 
            font-family: 'Roboto Mono', monospace; 
            background-color: var(--bg-dark); 
        }
        h1, h2 { font-family: 'Orbitron', sans-serif; }
        .neon-text-blue { color: var(--neon-blue); text-shadow: 0 0 5px var(--neon-blue); }
        .neon-glow-red { box-shadow: 0 0 10px rgba(255, 51, 102, 0.5); }
        .neon-border-blue { border-color: var(--neon-blue); box-shadow: 0 0 8px rgba(0, 255, 255, 0.3); }
        
        /* Custom File Input Styling */
        .custom-file-upload {
            background-color: var(--card-dark);
            border: 2px dashed #4b5563;
            transition: all 0.3s;
        }
        .custom-file-upload:hover {
            border-color: var(--neon-blue);
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.4);
        }
    </style>
</head>
<body class="text-gray-200 min-h-screen flex flex-col items-center p-4 sm:p-8">

    <div class="w-full max-w-2xl mx-auto bg-[var(--card-dark)] rounded-xl shadow-2xl overflow-hidden border-2 border-gray-700/50">

        <header class="p-6 text-center border-b border-gray-700/50">
            <h1 class="text-3xl sm:text-4xl font-black neon-text-blue tracking-wider">
                <span class="mr-2 text-4xl sm:text-5xl">⚡</span> NEURO-SCAN CLASSIFIER
            </h1>
            <p class="text-gray-400 mt-2 text-sm italic">
                PyTorch ResNet18 Analysis Protocol V.1.0
            </p>
        </header>

        <main class="p-6 space-y-8">
            <form method="POST" enctype="multipart/form-data" class="space-y-6">
                
                <div class="custom-file-upload flex flex-col items-center justify-center p-10 rounded-lg">
                    <label for="file-upload" class="cursor-pointer px-6 py-3 bg-[var(--neon-blue)] text-gray-900 font-bold rounded-full shadow-lg transition duration-300 transform hover:scale-105 neon-glow-blue hover:bg-white">
                        <svg class="w-6 h-6 inline-block mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12"></path></svg>
                        INITIATE X-RAY TRANSFER
                    </label>
                    <input id="file-upload" type="file" name="file" accept="image/*" required class="hidden" onchange="document.getElementById('file-name').innerText = this.files[0].name; document.getElementById('file-desc').classList.add('hidden');">
                    <p id="file-name" class="text-gray-400 mt-4 text-xs italic">Awaiting secure data input...</p>
                    <p id="file-desc" class="text-gray-500 mt-1 text-xs">Maximum resolution: 224x224 (Resized on server)</p>
                </div>

                <button type="submit" class="w-full px-4 py-3 bg-red-600 text-white font-extrabold rounded-lg shadow-xl hover:bg-[var(--neon-red)] transition duration-300 transform hover:-translate-y-0.5 focus:outline-none focus:ring-4 focus:ring-red-500 focus:ring-opacity-50 neon-glow-red">
                    EXECUTE DIAGNOSTIC SEQUENCE
                </button>
            </form>

            {% if prediction %}
            
                {% set is_pneumonia = prediction == 'Pneumonia' %}
                
                <div class="result mt-10 p-6 rounded-xl border-2 
                    {% if is_pneumonia %}
                        border-red-600 bg-red-900/20 neon-glow-red
                    {% else %}
                        border-green-600 bg-green-900/20 
                    {% endif %}
                    ">
                    
                    <div class="flex items-center justify-between border-b border-gray-600 pb-3 mb-3">
                        <h2 class="text-xl font-bold tracking-wider 
                            {% if is_pneumonia %}
                                text-[var(--neon-red)] animate-pulse
                            {% else %}
                                text-green-400
                            {% endif %}
                            ">
                            <span class="mr-2">{{ '🚨' if is_pneumonia else '🟢' }}</span> CLASSIFICATION REPORT
                        </h2>
                        <p class="text-sm text-gray-500 font-mono">{{ current_time }}</p>
                    </div>

                    <div class="flex flex-col sm:flex-row justify-between items-center space-y-4 sm:space-y-0 pt-2">
                        <div class="text-left">
                            <p class="text-sm text-gray-400 mb-1">FINAL DECISION:</p>
                            <p class="text-5xl font-extrabold 
                                {% if is_pneumonia %}
                                    text-[var(--neon-red)] tracking-widest
                                {% else %}
                                    text-green-400 tracking-wider
                                {% endif %}
                                ">
                                {{ prediction }}
                            </p>
                        </div>
                        
                        <div class="text-right">
                            <p class="text-sm text-gray-400 mb-1">CONFIDENCE INDEX:</p>
                            <p class="text-4xl font-extrabold 
                                {% if is_pneumonia %}
                                    text-[var(--neon-red)]
                                {% else %}
                                    text-green-400
                                {% endif %}
                                ">
                                {{ confidence }}%
                            </p>
                        </div>
                    </div>
                </div>
                
                <p class="text-xs text-gray-600 mt-6 italic text-center">
                    ATTENTION: This output is for screening purposes only. Consult medical personnel for final diagnosis.
                </p>

            {% endif %}

        </main>
    </div>

</body>
</html>
"""

# --- Flask Routes ---

@app.route("/", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if request.method == "POST":
        try:
            file = request.files['file']
            
            # 1. Image Preprocessing
            image = Image.open(io.BytesIO(file.read())).convert('RGB')
            img_tensor = transform(image).unsqueeze(0)

            with torch.no_grad():
                # 2. Model Prediction
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                
                # Get max probability and index
                max_prob, predicted_index = torch.max(probs, 1)
                
                # Extract numerical results
                predicted_class = CLASS_NAMES[predicted_index.item()]
                max_prob_value = max_prob.item() # The probability value (e.g., 0.95)
                confidence = round(max_prob_value * 100, 2)
                
                # 3. Apply the Confidence Threshold Rule (95%)
                if predicted_class == "Pneumonia" and max_prob_value < MIN_PNEUMONIA_CONFIDENCE:
                    # Override: If Pneumonia confidence is too low (< 95%), treat it as Normal.
                    prediction = "Normal"
                    # The original low confidence score is kept for display
                else:
                    # Use the model's direct prediction
                    prediction = predicted_class
                    
        except Exception as e:
            prediction = "SYSTEM FAILURE"
            confidence = 0
            print(f"Prediction error: {e}")

    # Render the styled template
    return render_template_string(
        form_html, 
        prediction=prediction, 
        confidence=confidence,
        current_time=current_time
    )

# --- Run App ---
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)