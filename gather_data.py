import cv2 as cv
import numpy as np
import mediapipe as mp

gesture = { 
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok'
} 

file = np.genfromtxt('gesture_train.csv', delimiter=',')

mp_hand = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hand = mp_hand.Hands(max_num_hands=1, static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FRAME_WIDTH, 600)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 600)

def click(event, x, y, flags, param):
    global data, file
    # 왼쪽 마우스 클릭 시, 해당 손 동작의 관절 사이 각도를 gesture_train.csv에 저장
    # 손 제스처는 1(one)으로 지정
    if event == cv.EVENT_LBUTTONDOWN:
        data = np.append(data, 1)
        file = np.vstack((file, data))  # 파일에 data를 이어 붙임  
        print("Gather Index finger")
        
    # 오른쪽 마우스 클릭 시, 해당 손 동작의 관절 사이 각도를 gesture_train.csv에 저장
    # 손 제스처는 0(fist)으로 지정
    elif event == cv.EVENT_RBUTTONDOWN:
        data = np.append(data, 0)
        file = np.vstack((file, data))  # 파일에 data를 이어 붙임  
        print("Gather Fist")
        
cv.namedWindow('Gather_Dataset')
cv.setMouseCallback('Gather_Dataset', click)

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("can't get frame")
        break

    res = hand.process(cv.cvtColor(frame, cv.COLOR_BGR2RGB))
    
    if res.multi_hand_landmarks:
        for landmarks in res.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, landmarks, mp_hand.HAND_CONNECTIONS,
                                       mp_styles.get_default_hand_landmarks_style(), mp_styles.get_default_hand_connections_style())
            
            joint = np.zeros((21, 3))
            for i, lm in enumerate(landmarks.landmark):
                joint[i] = [lm.x, lm.y, lm.z]
            
            v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:] # Parent knuckle
            v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:] # Child joint
            v = v2 - v1 
            
            v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

            angle = np.arccos(np.einsum('nt,nt->n',
                v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))

            angle = np.degrees(angle)

            data = np.array([angle], dtype=np.float32)      


    cv.imshow('Gather_Dataset', cv.flip(frame,1))
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
cv.waitKey(1)

# gesture_train.csv를 재저장 
np.savetxt('gesture_train.csv', file, delimiter=",") 