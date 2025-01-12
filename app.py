from flask import Flask, request, jsonify, send_from_directory, render_template
import os
import tensorflow as tf
import uuid
import cv2
from utils import process_image  # Simulación de procesamiento
from flask_cors import CORS
from deepface import DeepFace
import mediapipe as mp

# Configurar el entorno para TensorFlow
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Ocultar logs innecesarios de TensorFlow

# Crear la aplicación Flask
app = Flask(__name__)
CORS(app)

# Crear carpetas necesarias
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
imagenes_cliente_path = os.path.join(BASE_DIR, 'imagenesCliente')
processed_images_path = os.path.join(BASE_DIR, 'processed_images')
os.makedirs(imagenes_cliente_path, exist_ok=True)
os.makedirs(processed_images_path, exist_ok=True)

# Configuración de carpetas
app.config['IMAGENES_CLIENTE_FOLDER'] = imagenes_cliente_path
app.config['PROCESSED_IMAGES_FOLDER'] = processed_images_path

# Inicializar MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

# Función para redimensionar imágenes grandes con OpenCV
def resize_image(image_path, max_width=800):
    """
    Redimensiona una imagen si su ancho es mayor que max_width.
    """
    img = cv2.imread(image_path)
    height, width = img.shape[:2]
    if width > max_width:
        aspect_ratio = height / width
        new_width = max_width
        new_height = int(new_width * aspect_ratio)
        resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        cv2.imwrite(image_path, resized_img)

@app.route('/processed_images/<filename>')
def send_processed_image(filename):
    return send_from_directory(app.config['PROCESSED_IMAGES_FOLDER'], filename)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"error": "No se encontró ninguna imagen"}), 400

    file = request.files['image']
    unique_filename = f"{uuid.uuid4()}.jpg"
    temp_file_path = os.path.join(app.config['IMAGENES_CLIENTE_FOLDER'], unique_filename)

    # Guardar la imagen original temporalmente
    file.save(temp_file_path)

    # Redimensionar la imagen si es grande
    resize_image(temp_file_path)

    try:
        # Procesar la imagen con MediaPipe FaceMesh
        img = cv2.imread(temp_file_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return jsonify({"error": "No se detectaron rostros"}), 400

        # Analizar emociones con DeepFace
        emotion_analysis = DeepFace.analyze(temp_file_path, actions=['emotion'], enforce_detection=False)

        # Guardar las imágenes procesadas (sobrescribir siempre las mismas 4)
        processed_images = []
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            processed_image_name = f"processed_image_{idx}.jpg"
            processed_image_path = os.path.join(app.config['PROCESSED_IMAGES_FOLDER'], processed_image_name)

            # Dibujar los landmarks faciales en la imagen
            annotated_image = img.copy()
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

            cv2.imwrite(processed_image_path, annotated_image)
            processed_images.append(f"/processed_images/{processed_image_name}")

        return jsonify({
            "message": "Imágenes procesadas correctamente",
            "images": processed_images,
            "emotions": emotion_analysis
        })

    except Exception as e:
        return jsonify({"error": f"Error procesando la imagen: {str(e)}"}), 500

@app.route('/historico_imagenes')
def ver_historico():
    imagenes = os.listdir(app.config['IMAGENES_CLIENTE_FOLDER'])
    imagenes_urls = [f"/imagenesCliente/{img}" for img in imagenes if img.endswith('.jpg')]
    return jsonify({"imagenes": imagenes_urls})

@app.route('/imagenesCliente/<filename>')
def send_cliente_image(filename):
    return send_from_directory(app.config['IMAGENES_CLIENTE_FOLDER'], filename)

@app.route('/reprocesar', methods=['POST'])
def reprocesar_imagen():
    data = request.get_json()
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({"error": "No se proporcionó ninguna imagen"}), 400

    image_name = os.path.basename(image_url)
    image_path = os.path.join(app.config['IMAGENES_CLIENTE_FOLDER'], image_name)

    if not os.path.exists(image_path):
        return jsonify({"error": "La imagen no existe"}), 404

    try:
        # Procesar la imagen con MediaPipe FaceMesh
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(img_rgb)

        if not results.multi_face_landmarks:
            return jsonify({"error": "No se detectaron rostros"}), 400

        # Analizar emociones con DeepFace
        emotion_analysis = DeepFace.analyze(image_path, actions=['emotion'], enforce_detection=False)

        # Sobrescribir siempre las mismas imágenes procesadas
        processed_images = []
        for idx, face_landmarks in enumerate(results.multi_face_landmarks):
            processed_image_name = f"processed_image_{idx}.jpg"
            processed_image_path = os.path.join(app.config['PROCESSED_IMAGES_FOLDER'], processed_image_name)

            # Dibujar los landmarks faciales en la imagen
            annotated_image = img.copy()
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * img.shape[1])
                y = int(landmark.y * img.shape[0])
                cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

            cv2.imwrite(processed_image_path, annotated_image)
            processed_images.append(f"/processed_images/{processed_image_name}")

        return jsonify({
            "message": "Imagen reprocesada correctamente",
            "images": processed_images,
            "emotions": emotion_analysis
        })

    except Exception as e:
        return jsonify({"error": f"Error reprocesando la imagen: {str(e)}"}), 500

# Ruta para eliminar una imagen
@app.route('/eliminar_imagen', methods=['POST'])
def eliminar_imagen():
    data = request.get_json()
    image_url = data.get('imageUrl')

    if not image_url:
        return jsonify({'success': False, 'message': 'No se proporcionó una URL de imagen'}), 400

    imagenes_cliente_path = os.path.join(app.config['IMAGENES_CLIENTE_FOLDER'], os.path.basename(image_url))

    if not os.path.exists(imagenes_cliente_path):
        return jsonify({'success': False, 'message': 'La imagen no existe en el servidor'}), 404

    try:
        os.remove(imagenes_cliente_path)
        return jsonify({'success': True, 'message': 'Imagen eliminada con éxito'}), 200
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)}), 500

# Ejecuta la aplicación Flask
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
