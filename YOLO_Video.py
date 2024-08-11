from ultralytics import YOLO
import cv2
import math
#menyiapkan objek untuk membaca video
def video_detection(path_x, model_path, class_names):
    video_capture = path_x
    cap = cv2.VideoCapture(video_capture)

    model = YOLO(model_path) #memuat model yolo
    while True:
        success, img = cap.read()
        if not success:
            break
        results = model(img, stream=True)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]#jordinat boundingbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)#Mengonversi koordinat bounding box dari tipe float ke tipe integer
                print(x1, y1, x2, y2) #Mencetak koordinat bounding box untuk debugging atau informasi.
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3) #warna boundingbox
                conf = math.ceil((box.conf[0] * 100)) / 100 #Mengambil nilai kepercayaan (confidence)
                cls = int(box.cls[0]) # mengambil nama kelas dari daftar class_names.
                class_name = class_names[cls] #Mengambil nama kelas yang sesuai
                label = f'{class_name} {conf}' #Membuat label teks
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0] #Menghitung ukuran teks label 
                print(t_size) #Menghitung ukuran
                c2 = x1 + t_size[0], y1 - t_size[1] - 3 
                cv2.rectangle(img, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  #Menggambar kotak latar belakang untuk teks label di atas bounding box objek
                cv2.putText(img, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA) #Menulis teks label di atas gambar
        yield img

    cap.release()

cv2.destroyAllWindows()
