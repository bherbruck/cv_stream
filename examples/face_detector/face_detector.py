from cv_stream import Camera, CameraServer
import cv2


classifier = cv2.CascadeClassifier('haar_cascade/face.xml')


def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    objects = [(x, y, w, h)
               for x, y, w, h
               in classifier.detectMultiScale(gray, 1.3, 5)]

    for obj in objects:
        cv2.rectangle(frame,
                      pt1=(obj[0], obj[1]),
                      pt2=(obj[0] + obj[2], obj[1] + obj[3]),
                      color=(255, 0, 0),
                      thickness=2)

    return frame


camera = Camera(frame_callback=process_frame)
server = CameraServer(camera=camera, password='123')
server.run()
