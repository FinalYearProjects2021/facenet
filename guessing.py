from PIL import Image
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from matplotlib import pyplot
from numpy import load
from numpy import expand_dims
from numpy import asarray
from numpy import savez_compressed
from keras.models import load_model
import numpy as np
from numpy import load
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import joblib


detector = MTCNN()
face_net = load_model('facenet_keras.h5')
model = joblib.load('final_model.sav')
in_encoder = Normalizer(norm='l2')

print("Finally Loaded")

#model = joblib.load('final_model.sav')

sh = un = sm = sk = 0


def draw_face(filename, name, required_size=(160, 160)):
    #    image = Image.open(filename)

    #   image = image.convert('RGB')

    # pixels = asarray(filename)

    results = detector.detect_faces(filename)
    # print("results:{}".format(results))

    for face in results:
        x, y, wi, he = face['box']
        cv2.rectangle(filename, (x, y), (x + wi, y + he), (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(filename, name, (x, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
    image2 = cv2.cvtColor(filename, cv2.COLOR_RGB2BGR)
    return image2


def extract_face(filename, required_size=(160, 160)):
    #    image = Image.open(filename)

    #   image = image.convert('RGB')

    pixels = asarray(filename)

    results = detector.detect_faces(filename)
    # print("results:{}".format(results))

    image2 = cv2.cvtColor(filename, cv2.COLOR_RGB2BGR)
    if len(results) != 0:
        x1, y1, width, height = results[0]['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face = pixels[y1:y2, x1:x2]
        image = Image.fromarray(face)
        image = image.resize(required_size)
        face_array = asarray(image)

        return [face_array, image2]
    else:
        return [0, image2]


import cv2


# face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = face_net.predict(samples)
    return yhat[0]


cap = cv2.VideoCapture(0)
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_array, frame = extract_face(image)
    if type(face_array) != type(0):
        # print(type(face_array))
        # print(face_array.shape)
        # m,n,o = face_array.shape
        # face_array = np.reshape(face_array,(1,m,n,o))
        # print(type(face_array))
        newTrainX = list()
        # face_array = face_array.astype('float32')
        embedding = get_embedding(model, face_array)
        # print(type(face_array))
        # print(face_array.shape)
        newTrainX.append(embedding)
        newTrainX = asarray(newTrainX)
        trainX = in_encoder.transform(newTrainX)
        # print(type(trainX))
        # print(trainX.shape)
        # print(len(trainX))
        # print("hello")
        # print("shape:{}:{}".format(trainX.shape,type(trainX)))

        # samples = expand_dims(trainX, axis=0)
        #        print("shape:{}:{}".format(samples.shape,type(samples)))
        y_class = model.predict(trainX)
        # print(y_class)
        y_prob = model.predict_proba(trainX)
        # print(y_prob)
        class_index = y_class[0]
        print(class_index)
        class_probability = y_prob[0, class_index] * 100
        print(class_probability)
        if class_probability < 96.00:
            name = "Unknown"
            un += 1
        else:
            if class_index == 0:
                name = "shashank"
                sh += 1
            elif class_index == 1:
                name = "skanda"
                sk += 1
		 else:
			name = "sumanth"	

        frame = draw_face(image, name)

    cv2.imshow('shashank', frame)
    if (cv2.waitKey(1) & 0xff == ord('q')):
        break
    if (sh == 25) or (sm == 25) or (un == 25) or (sk == 25):
        break
cap.release()
cv2.destroyAllWindows()
if sh == 25:
    print("The person is Shashank")
elif sm == 25:
    print("The person is Sumanth")
elif sk == 25:
    print("skanda")
else:
    print("unknown person")	