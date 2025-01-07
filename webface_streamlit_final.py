import streamlit as st
import cv2
import numpy as np
import pyodbc
import os
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# Initialize FaceAnalysis
app = FaceAnalysis(det_size=(640, 640))
app.prepare(ctx_id=0)  # Use GPU if available; set ctx_id=-1 for CPU

# Database setup variables
server = '20.244.109.231,1433'  # Update with your SQL server address
database = 'test'  # Update with your database name
username = 'vmukti'  # Update with your SQL username
password = 'bhargav@123456'  # Update with your SQL password

# Database connection
conn_str = f'DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};UID={username};PWD={password}'
conn = pyodbc.connect(conn_str)
cursor = conn.cursor()

# Create the Embeddings table if it doesn't exist
cursor.execute('''
    IF NOT EXISTS (SELECT * FROM sysobjects WHERE name='Embeddings' and xtype='U')
    CREATE TABLE Embeddings (
        PersonName NVARCHAR(100) PRIMARY KEY,
        Embedding VARBINARY(MAX)
    );
''')
conn.commit()

# Load embeddings from the database
def load_embeddings_from_sql():
    cursor.execute("SELECT PersonName, Embedding FROM Embeddings")
    known_faces = {}
    for row in cursor.fetchall():
        person_name = row[0]
        embedding = np.frombuffer(row[1], dtype=np.float32)
        known_faces[person_name] = embedding
    return known_faces

known_faces_db = load_embeddings_from_sql()

# Find the best match for a face
def find_best_match(embedding, known_faces_db, threshold=0.5):
    best_match = None
    best_similarity = -1

    for name, known_embedding in known_faces_db.items():
        similarity = cosine_similarity([embedding], [known_embedding])[0][0]
        if similarity > best_similarity and similarity >= threshold:
            best_match = name
            best_similarity = similarity

    return best_match, best_similarity

# Process a single frame
def process_frame(image):
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image_rgb)

    for face in faces:
        bbox = face.bbox.astype(int)
        embedding = face.embedding
        match_name, similarity = find_best_match(embedding, known_faces_db)
        label = f"{match_name}" if match_name else "Unknown"
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        cv2.putText(image, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return image

# Function to extract face embedding from an image
def get_face_embedding(image_path):
    img = cv2.imread(image_path)
    faces = app.get(img)

    if len(faces) == 0:
        return None  # No face detected

    return faces[0].embedding

# Function to save embeddings to SQL Server
def save_embeddings_to_sql(person_name, embedding):
    embedding_binary = embedding.tobytes()

    cursor.execute("""
        IF EXISTS (SELECT 1 FROM Embeddings WHERE PersonName = ?)
        UPDATE Embeddings SET Embedding = ? WHERE PersonName = ?
        ELSE
        INSERT INTO Embeddings (PersonName, Embedding) VALUES (?, ?)
    """, (person_name, embedding_binary, person_name, person_name, embedding_binary))

    conn.commit()
    return f"Embedding for {person_name} saved successfully!"

# Function to retrieve embedding from SQL Server
def load_embedding_from_sql(person_name):
    cursor.execute("SELECT Embedding FROM Embeddings WHERE PersonName = ?", person_name)
    row = cursor.fetchone()

    if row:
        embedding_binary = row[0]
        embedding = np.frombuffer(embedding_binary, dtype=np.float32)
        return f"Embedding for {person_name} loaded successfully!", embedding
    else:
        return f"No embedding found for {person_name}", None

# Function to get all registered names
def get_all_registered_faces():
    cursor.execute("SELECT PersonName FROM Embeddings")
    rows = cursor.fetchall()

    if rows:
        names = [[row[0]] for row in rows]
        return names
    else:
        return [["No registered faces found"]]

# Function to delete a face by name from SQL Server
def remove_face_from_sql(person_name):
    cursor.execute("DELETE FROM Embeddings WHERE PersonName = ?", person_name)
    conn.commit()

    return f"Embedding for {person_name} removed successfully!" if cursor.rowcount > 0 else f"No face found for {person_name}"

# Function to open a camera by index
def open_camera_by_index():
    for index in range(3):  # Try video0, video1, video2
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            return cap
    return None

# Streamlit UI for live webcam detection and embedding management
def main():
    st.title("Live Face Detection and Recognition")

    # Tabs for functionality
    tab1, tab2, tab3, tab4 = st.tabs(["Live Detection", "Register Face", "Retrieve Embedding", "View/Remove Faces"])

    with tab1:
        st.subheader("Live Detection")
        run = st.checkbox("Start Webcam")
        FRAME_WINDOW = st.image([])

        cap = open_camera_by_index()
        if cap is None:
            st.error("No available webcam found.")

        while run and cap is not None:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to access webcam.")
                break

            processed_frame = process_frame(frame)
            FRAME_WINDOW.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")

        if cap:
            cap.release()

    with tab2:
        st.subheader("Register Face")
        person_name = st.text_input("Enter Person Name:")

        option = st.radio("Select input method:", ("Upload an Image", "Use Webcam"))

        if option == "Upload an Image":
            uploaded_file = st.file_uploader("Upload an Image:", type=["jpg", "png", "jpeg"])

            if st.button("Save Embedding", key="upload"):
                if uploaded_file and person_name:
                    temp_image_path = f"temp_{uploaded_file.name}"
                    with open(temp_image_path, "wb") as f:
                        f.write(uploaded_file.read())

                    embedding = get_face_embedding(temp_image_path)
                    if embedding is not None:
                        result = save_embeddings_to_sql(person_name, embedding)
                        st.success(result)
                    else:
                        st.error("No face detected in the image.")

                    os.remove(temp_image_path)
                else:
                    st.error("Please provide both a name and an image.")

        elif option == "Use Webcam":
            st.write("Press 'Capture' to take a photo.")
            cap = open_camera_by_index()

            if cap is None:
                st.error("No available webcam found.")

            captured_image_slot = st.empty()

            if st.button("Capture") and cap is not None:
                ret, frame = cap.read()
                if ret:
                    captured_image_slot.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")
                    faces = app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if faces:
                        embedding = faces[0].embedding
                        if embedding is not None:
                            result = save_embeddings_to_sql(person_name, embedding)
                            st.success(result)
                        else:
                            st.error("Could not extract an embedding from the detected face.")
                    else:
                        st.error("No face detected in the captured photo.")

                cap.release()

    with tab3:
        st.subheader("Retrieve Embedding")
        person_name = st.text_input("Enter Person Name to Retrieve:", key="retrieve")

        if st.button("Retrieve"):
            result, embedding = load_embedding_from_sql(person_name)
            st.write(result)

    with tab4:
        st.subheader("View or Remove Registered Faces")
        st.write("### Registered Faces")
        faces = get_all_registered_faces()
        st.table(faces)

        person_name = st.text_input("Enter Person Name to Remove:", key="remove")
        if st.button("Remove"):
            result = remove_face_from_sql(person_name)
            st.write(result)

if __name__ == "__main__":
    main()

