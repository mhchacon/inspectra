from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Certifique-se de que a pasta de uploads existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])


class Detection:
    def __init__(self):
        # Baixe os pesos do YOLO e altere o caminho conforme necessário
        self.model = YOLO(r"yolov11_custom.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                # Obtém a classe do objeto detectado
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # Define a cor do quadrado com base no status do objeto
                if class_name == "Intacto":  # Substitua "intact" pelo nome da classe intacta no seu modelo
                    box_color = (0, 255, 0)  # Verde para objetos intactos
                else:
                    box_color = (255, 0, 0)  # Vermelho para outros objetos

                # Desenha o quadrado ao redor do objeto
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                              (int(box.xyxy[0][2]), int(box.xyxy[0][3])), box_color, rectangle_thickness)

                # Adiciona o nome da classe acima do quadrado
                cv2.putText(img, f"{class_name}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, box_color, text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img

    def detect_from_video_frame(self, frame):
        result_img, _ = self.predict_and_detect(frame, classes=[], conf=0.5)
        return result_img


detection = Detection()


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = Image.open(file_path).convert("RGB")
        img = np.array(img)
        img = cv2.resize(img, (512, 512))
        img = detection.detect_from_image(img)
        output = Image.fromarray(img)

        buf = io.BytesIO()
        output.save(buf, format="PNG")
        buf.seek(0)

        os.remove(file_path)
        return send_file(buf, mimetype='image/png')


@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona o frame para 512x512
        frame = cv2.resize(frame, (512, 512))

        # Aplica a detecção de objetos no frame
        frame = detection.detect_from_video_frame(frame)

        # Converte o frame para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Converte o buffer para bytes
        frame_bytes = buffer.tobytes()

        # Retorna o frame no formato de streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Libera a câmera quando o loop termina
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8000, debug=True)