import cv2
import numpy as np

net = cv2.dnn.readNetFromDarknet(r"C:\Users\jlukas\Desktop\My Projects\MachineLearning\Custom_Train_Model-Detect_Hand_Gesture-Final\Final_Custom_Working_Model_DetectMy_Hand_Yolov3-Tiny\yolov3-tiny_myhand.cfg", r"C:\Users\jlukas\Desktop\My Projects\MachineLearning\Custom_Train_Model-Detect_Hand_Gesture-Final\Final_Custom_Working_Model_DetectMy_Hand_Yolov3-Tiny\yolov3-tiny_myhand_last.weights")

## With CPU
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

## With CUDA
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

classes = ['right', 'left', 'stop', 'up', 'down']
cap = cv2.VideoCapture(0)
whT = 320
confThreshold =0.5
nmsThreshold= 0.2

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = []
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2] * wT), int(det[3] * hT)
                x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
        cv2.putText(img, f'{classes[classIds[i]].upper()} {int(confs[i] * 100)}%',
                   (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

while True:
    sucess, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)

    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1]) for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
