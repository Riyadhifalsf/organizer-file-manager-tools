import os
import shutil
import pickle
import numpy as np
from PIL import Image, ExifTags
from sklearn.ensemble import RandomForestClassifier

# ================= STORAGE =================
MODEL_DIR = r"D:\ai_data"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
DATASET_PATH = os.path.join(MODEL_DIR, "dataset.pkl")

# ================= CONFIG =================
SRC_FOLDER = r"D:\scan"
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
    return RandomForestClassifier(n_estimators=120)

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

# ================= IMAGE FEATURE =================
def img_to_vector(path):
    try:
        img = Image.open(path).resize(THUMB_SIZE).convert("RGB")
        return np.array(img).flatten() / 255.0
    except:
        return None

# ================= EXIF =================
def get_exif_device(path):
    try:
        img = Image.open(path)
        exif = img._getexif()
        if not exif:
            return None

        data = {}
        for tag, value in exif.items():
            name = ExifTags.TAGS.get(tag, tag)
            data[name] = str(value)

        make = data.get("Make", "").lower()
        model = data.get("Model", "").lower()

        device = f"{make} {model}".strip()

        if "iphone" in device:
            return "iPhone"
        if "samsung" in device:
            return "Samsung"
        if "xiaomi" in device or "redmi" in device:
            return "Xiaomi"
        if "oppo" in device:
            return "Oppo"
        if "vivo" in device:
            return "Vivo"

        return None
    except:
        return None

# ================= SMART DETECTION =================
def detect_visual_type(path):
    """
    Heuristik sederhana:
    - Screenshot: banyak warna flat + kontras tinggi + teks
    - Camera: warna lebih natural & bervariasi
    """
    try:
        img = Image.open(path).resize((64, 64)).convert("L")  # grayscale
        arr = np.array(img)

        variance = np.var(arr)
        edges = np.mean(np.abs(np.diff(arr)))

        # threshold sederhana
        if variance < 500 and edges > 20:
            return "Screenshots"
        else:
            return "Camera"

    except:
        return "Unknown"

# ================= INIT =================
model = load_model()
X_data, y_data = load_dataset()

if len(X_data) > 0:
    try:
        model.fit(X_data, y_data)
    except:
        X_data, y_data = [], []

# ================= PREDICT =================
def predict(file):
    name = os.path.basename(file).lower()

    # ===== RULE NAMA FILE =====
    if "whatsapp" in name:
        return "WhatsApp", 1.0

    if "screenshot" in name or "screen" in name:
        return "Screenshots", 1.0

    # ===== EXIF DEVICE =====
    device = get_exif_device(file)
    if device:
        return "Camera", 0.9

    # ===== VISUAL DETECTION =====
    visual = detect_visual_type(file)
    if visual != "Unknown":
        return visual, 0.7

    # ===== AI =====
    vec = img_to_vector(file)
    if vec is None or len(X_data) < 20:
        return "Unknown", 0

    probs = model.predict_proba([vec])[0]
    idx = np.argmax(probs)

    return model.classes_[idx], probs[idx]

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
    print("🚀 AI ORGANIZER V2 START\n")

    files = scan_files()
    total = len(files)

    print(f"📂 Total file: {total}\n")

    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{total}] 📄 {file}")

        pred, conf = predict(file)
        print(f"🤖 Prediksi: {pred} ({conf:.2f})")

        # ===== AUTO =====
        if conf >= 0.85:
            label = pred
            print("⚡ Auto")
        else:
            label = input("✏️ Kategori: ").strip()
            if not label:
                label = "Unknown"

        try:
            new_path = move_file(file, label)
            print(f"➡️ {new_path}")
        except Exception as e:
            print("❌ ERROR MOVE:", e)
            continue

        learn(new_path, label)

        if i % 10 == 0:
            save_all(model, X_data, y_data)

    save_all(model, X_data, y_data)

    print("\n🔥 SELESAI — AI MAKIN PINTER")

# ================= RUN =================
if __name__ == "__main__":
    main()