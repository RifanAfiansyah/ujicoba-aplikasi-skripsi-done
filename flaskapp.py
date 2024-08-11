from flask import Flask, render_template, Response, session, url_for
from YOLO_Video import video_detection
import cv2

app = Flask(__name__)

def generate_frames_web(path_x, model_path, class_names):
    for detection_ in video_detection(path_x, model_path, class_names):
        ref, buffer = cv2.imencode('.jpg', detection_)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/silat", methods=['GET'])
def silat():
    video_url = url_for('webapp_silat')
    return render_template('deteksi.html', video_url=video_url)

@app.route('/taekwondo', methods=['GET'])
def taekwondo():
    video_url = url_for('webapp_taekwondo')
    return render_template('deteksi.html', video_url=video_url)

@app.route('/karate', methods=['GET'])
def karate():
    video_url = url_for('webapp_karate')
    return render_template('deteksi.html', video_url=video_url)

@app.route('/webapp/silat')
def webapp_silat():
    silat_classes = ['kuda kuda belakang', 'kuda kuda depan', 'kuda kuda samping', 'kuda kuda silang belakang', 'kuda kuda silang depan', 'kuda kuda tengah']
    return Response(generate_frames_web(path_x=0, model_path='YOLO_Weights/silat.pt', class_names=silat_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp/taekwondo')
def webapp_taekwondo():
    taekwondo_classes = ['ap kubi', 'ap seogi', 'apkoa seogi', 'beom seogi', 'dwi kubi', 'dwikkoa seogi', 'juchum seogi', 'moa seogi', 'naranhi seogi', 'wen seogi']  
    return Response(generate_frames_web(path_x=0, model_path='YOLO_Weights/taekwondo.pt', class_names=taekwondo_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/webapp/karate')
def webapp_karate():
    karate_classes = ['fudo dachi', 'heisoku dachi', 'kiba dachi', 'kokutsu dachi', 'kosa dachi', 'musubi dachi', 'neko ashi dachi', 'renoji dachi', 'zenkutsu dachi']  
    return Response(generate_frames_web(path_x=0, model_path='YOLO_Weights/karate.pt', class_names=karate_classes), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
