import cv2
import mediapipe
capture = cv2.VideoCapture(0)

mediapipehand = mediapipe.solutions.hands
tangan = mediapipehand.Hands()

while True :
    success, img = capture.read()
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = tangan.process(imgRGB)
    if results.multi_hand_landmarks:
        print("tangan")
    else :
        print("tidak ada")
    cv2.imshow ("webcam",img)
    cv2.waitKey(10)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
capture.release()
cv2.destroyAllWindows()