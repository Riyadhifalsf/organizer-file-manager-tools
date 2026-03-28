import os
import shutil
import pickle
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# ================= STORAGE =================
MODEL_DIR = r"D:\ai_data"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
DATASET_PATH = os.path.join(MODEL_DIR, "dataset.pkl")

# ================= CONFIG =================
SRC_FOLDER = r"D:\output\Pictures\training"
DST_FOLDER = r"D:\output\Learning"

os.makedirs(DST_FOLDER, exist_ok=True)

IMAGE_EXT = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".jfif")
THUMB_SIZE = (32, 32)

SKIP_FOLDERS = ["thumbnails", "cache", "temp", "$recycle.bin"]

# ================= LOAD =================
def load_model():
    if os.path.exists(MODEL_PATH):
        try:
            return pickle.load(open(MODEL_PATH, "rb"))
        except:
            print("⚠️ Model rusak, reset...")
    return RandomForestClassifier(n_estimators=100)

def load_dataset():
    if os.path.exists(DATASET_PATH):
        try:
            return pickle.load(open(DATASET_PATH, "rb"))
        except:
            print("⚠️ Dataset rusak, reset...")
    return [], []

def save_all(model, X, y):
    try:
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(model, f)

        with open(DATASET_PATH, "wb") as f:
            pickle.dump((X, y), f)
    except Exception as e:
        print("❌ ERROR SAVE:", e)

# ================= FEATURE =================
def img_to_vector(path):
    try:
        img = Image.open(path).resize(THUMB_SIZE).convert("RGB")
        return np.array(img).flatten() / 255.0
    except:
        return None

# ================= INIT =================
model = load_model()
X_data, y_data = load_dataset()

if len(X_data) > 0:
    try:
        model.fit(X_data, y_data)
    except:
        X_data, y_data = [], []

# ================= RULE + AI =================
def predict(file):
    name = os.path.basename(file).lower()

    # 🔥 RULE BASED (PRIORITAS TINGGI)
    if "whatsapp" in name:
        return "WhatsApp", 1.0

    if "screenshot" in name or "screen" in name:
        return "Screenshots", 1.0

    if name.startswith(("img_", "img", "camera")):
        return "Camera", 0.85

    # ================= AI =================
    vec = img_to_vector(file)

    if vec is None or len(X_data) < 10:
        return "Unknown", 0

    try:
        probs = model.predict_proba([vec])[0]
        idx = np.argmax(probs)
        return model.classes_[idx], probs[idx]
    except:
        return "Unknown", 0

# ================= LEARN =================
def learn(file, label):
    global X_data, y_data, model

    vec = img_to_vector(file)
    if vec is None:
        return

    X_data.append(vec)
    y_data.append(label)

    try:
        model.fit(X_data, y_data)
    except:
        return

# ================= MOVE =================
def move_file(src, category):
    dest_folder = os.path.join(DST_FOLDER, category)
    os.makedirs(dest_folder, exist_ok=True)

    base = os.path.basename(src)
    new_path = os.path.join(dest_folder, base)

    counter = 1
    while os.path.exists(new_path):
        name, ext = os.path.splitext(base)
        new_path = os.path.join(dest_folder, f"{name}_{counter}{ext}")
        counter += 1

    shutil.move(src, new_path)
    return new_path

# ================= SCAN =================
def scan_files():
    files = []

    for root, dirs, fs in os.walk(SRC_FOLDER):
        if any(skip in root.lower() for skip in SKIP_FOLDERS):
            continue

        for f in fs:
            if f.lower().endswith(IMAGE_EXT):
                files.append(os.path.join(root, f))

    return files

# ================= MAIN =================
def main():
    print("🔍 Scanning...\n")

    files = scan_files()
    total = len(files)

    print(f"📂 Total file: {total}\n")

    if not files:
        return

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{total}] 📄 {file}")

        pred, conf = predict(file)
        print(f"🤖 Prediksi: {pred} ({conf:.2f})")

        # 🔥 AUTO MODE
        if conf > 0.9:
            label = pred
            print("⚡ Auto AI")
        else:
            label = input("✏️ Kategori (enter=Unknown): ").strip()
            if not label:
                label = "Unknown"

        try:
            new_path = move_file(file, label)
            print(f"➡️ Pindah ke: {new_path}")
        except Exception as e:
            print("❌ ERROR MOVE:", e)
            continue

        learn(new_path, label)

        # 💾 SAVE tiap 10 file biar ringan
        if i % 10 == 0:
            save_all(model, X_data, y_data)

    # save terakhir
    save_all(model, X_data, y_data)

    print("\n🔥 SELESAI — AI SUDAH BELAJAR")

# ================= RUN =================
if __name__ == "__main__":
    main()