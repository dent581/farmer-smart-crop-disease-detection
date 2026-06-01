#!/bin/bash
# ═══════════════════════════════════════════════
# FarmAdvisor AI — One-Click Setup for macOS
# ═══════════════════════════════════════════════

echo ""
echo "🌿 FarmAdvisor AI — Setup Script"
echo "══════════════════════════════════"

# Step 1: Find the right python (the one with packages)
PYTHON=""
for cmd in python3.9 /Library/Developer/CommandLineTools/usr/bin/python3 python3 python; do
    if command -v $cmd &>/dev/null; then
        VER=$($cmd --version 2>&1)
        echo "Found: $cmd → $VER"
        PYTHON=$cmd
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "❌ No Python found. Install from https://python.org"
    exit 1
fi

# Step 2: Create virtual environment
echo ""
echo "📦 Creating virtual environment..."
$PYTHON -m venv venv
source venv/bin/activate

# Step 3: Install all packages inside venv
echo ""
echo "📥 Installing packages (this may take 2-3 minutes)..."
pip install --upgrade pip --quiet
pip install flask tensorflow numpy pillow scikit-learn werkzeug --quiet
echo "✅ Packages installed"

# Step 4: Create demo model
echo ""
echo "🤖 Creating demo CNN model..."
python create_demo.py

# Step 5: Train NLP model
echo ""
echo "💬 Training NLP text model..."
python train_text_model.py

# Step 6: Launch app
echo ""
echo "🚀 Starting FarmAdvisor AI..."
echo "   Open http://127.0.0.1:5000 in your browser"
echo ""
python app.py
