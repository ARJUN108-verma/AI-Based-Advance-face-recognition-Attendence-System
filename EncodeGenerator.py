import cv2
import face_recognition
import pickle
import os
from supabase import create_client, Client

# Supabase Configuration
SUPABASE_URL = "https://auqxgnhnolllmubcngtt.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImF1cXhnbmhub2xsbG11YmNuZ3R0Iiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc2MjA2NTUzNiwiZXhwIjoyMDc3NjQxNTM2fQ.5xR_H-ijwflGV8VJ0MO8UdKDYKpAgMVkSHT0ZuaVbMA"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
print("✅ Supabase connected successfully (EncodeGenerator)")

# Folder containing student images
folderPath = 'Images'
pathList = os.listdir(folderPath)
print(f"📁 Found {len(pathList)} images: {pathList}")

imgList = []
studentIds = []

# Load images and extract student IDs
for file_name in pathList:
    file_path = os.path.join(folderPath, file_name)

    # Read image
    img = cv2.imread(file_path)
    if img is None:
        print(f"⚠️ Could not read image: {file_name}")
        continue

    imgList.append(img)
    studentIds.append(os.path.splitext(file_name)[0])

print(f"✅ Loaded {len(imgList)} images for encoding.")

def findEncodings(imagesList):
    encodeList = []
    for idx, img in enumerate(imagesList):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img_rgb)
        if len(encodes) == 0:
            print(f"⚠️ No face found in image: {studentIds[idx]}")
            continue
        encodeList.append(encodes[0])
        print(f"✅ Encoded: {studentIds[idx]}")
    return encodeList

print("⚙️ Encoding Started ...")
encodeListKnown = findEncodings(imgList)
print(f"✅ Encoding Complete. Total encodings: {len(encodeListKnown)}")

encodeListKnownWithIds = [encodeListKnown, studentIds]

# Save locally
with open("EncodeFile.p", 'wb') as file:
    pickle.dump(encodeListKnownWithIds, file)
print("💾 Encode file saved as EncodeFile.p")

# Optional: Upload images to Supabase (only if encoding succeeded)
for file_name in pathList:
    file_path = os.path.join(folderPath, file_name)
    with open(file_path, "rb") as f:
        res = supabase.storage.from_("student-images").update(file_name, f)
        print(f"📤 Uploaded: {file_name} → {res}")