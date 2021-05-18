import cv2

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def extract_image(img):
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray,1.3,5)
    print(faces)
    if faces is not ():
        return img

cap = cv2.VideoCapture(0)
count = 0


while cap.isOpened():
    ret,frame = cap.read()
    if extract_image(frame) is not None:
        count += 1
        #face = cv2.resize(extract_image(frame),(400,400))
        face = extract_image(frame)
        file_path = './images/'+str(count)+'.jpg'
        cv2.imwrite(file_path,face)
        frame = cv2.putText(face,str(count),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),2)

    cv2.imshow('Capture_pictures',frame)
    if (cv2.waitKey(1) & 0xff == ord('q'))or(count == 30):
        break
cap.release()
cv2.destroyAllWindows()