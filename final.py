import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
from PyQt5.QtWidgets import *

capture_running = False  # cv.VideoCapture()의 실행 여부
capture_save = False  # 사진 저장 여부. 'Save' 버튼을 누르면, True가 됨
BLACK = (0, 0, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
color = BLACK  # 선의 기본 색을 BLACK으로 지정

index_finger_history = []  # 현재 손으로 그리고 있는 위치를 담는 변수 
before_index_finger_history = []  # 이전에 손으로 그렸던 위치를 담는 변수

class DrawByFinger(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Draw by finger')
        self.setGeometry(800, 100, 500, 100)
        
        startBtn = QPushButton('Start Draw', self)  # 손으로 그림 그리기 시작 버튼
        changeColorBtn = QPushButton('Change Color', self)  # 선의 색깔을 바꾸는 버튼
        self.pickCombo = QComboBox(self)
        self.pickCombo.addItems(['black', 'red', 'yellow'])
        exitBtn = QPushButton('Exit', self)  # 종료 버튼
        capturBtn = QPushButton('Save', self)  # 사진 저장 버튼
        eraseBtn = QPushButton('Erase', self)  # 그림 지우기 버튼
        
        startBtn.setGeometry(10, 10, 120, 30)
        changeColorBtn.setGeometry(130, 10, 120, 30)
        self.pickCombo.setGeometry(130, 45, 120, 30)
        eraseBtn.setGeometry(250, 10, 120, 30)
        capturBtn.setGeometry(250, 45, 120, 30)
        exitBtn.setGeometry(390, 10, 100, 30)
        
        startBtn.clicked.connect(startBtnAction)
        changeColorBtn.clicked.connect(self.changeColorAction)
        eraseBtn.clicked.connect(self.eraseBtnAction)
        capturBtn.clicked.connect(self.saveAction)
        exitBtn.clicked.connect(self.exitBtnAction)
        
    def changeColorAction(self):
        global color
        pick_color = self.pickCombo.currentIndex()
        if pick_color == 0:
            color = BLACK
        elif pick_color == 1:
            color = RED
        elif pick_color == 2:
            color = YELLOW
        
    # 비디오를 캡쳐하여 사진을 저장할 수 있음
    # capture_save 변수를 True로 바꿔 176번 줄에서 사진을 저장할 수 있도록 구현함
    def saveAction(self):
        global capture_save
        capture_save = True
    
    # 지금까지 그렸던 모든 그림을 지움  
    def eraseBtnAction(self):
        index_finger_history.clear()
        before_index_finger_history.clear()
    
    # cv.VideoCapture()와 PyQt5 모두 종료    
    def exitBtnAction(self):
        global capture_running
        capture_running = False
        self.close()

# gesture_train.csv가 가지고 있는 손 제스처 label 목록
gesture = {
    0:'fist', 1:'one', 2:'two', 3:'three', 4:'four', 5:'five',
    6:'six', 7:'rock', 8:'spiderman', 9:'yeah', 10:'ok',
}

# gesture_train.csv 에는 각 제스처의 
# 손가락 마디끼리의 각도 정보가 담겨져 있음
# 이 데이터들을 knn으로 학습시켜 손 제스처 판단
file = np.genfromtxt('gesture_train_if.csv', delimiter=',')
angle = file[:,:-1].astype(np.float32)
label = file[:, -1].astype(np.float32)
knn = cv.ml.KNearest_create()
knn.train(angle, cv.ml.ROW_SAMPLE, label)

max_num_hands = 1  # 최대 한 손만 인식하도록 설정
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=max_num_hands,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
       
def draw_line(frame, positions):
    for i in range(len(positions)-1):
        cv.line(frame, positions[i][0], positions[i+1][0], positions[i][1], 2)

def startBtnAction():
    global capture_running
    global capture_save
    capture_running = True
    save_picture_num = 0  # 저장된 사진 파일 번호. 사진 파일 이름에 사용
    
    cap = cv.VideoCapture(0)
    
    while capture_running:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame = cv.flip(frame, 1)  # 비디오 좌우 반전
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame_height, frame_weight, _ = frame.shape

        result = hands.process(frame)
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)  
            
        if result.multi_hand_landmarks is not None:
            for res in result.multi_hand_landmarks:
                
                # mediapipe의 hand landmark model이 검출한 손가락 관절의 
                # x, y, z 좌표를 knuckle 리스트에 저장
                knuckle = np.zeros((21, 3))
                for i, lm in enumerate(res.landmark):
                    knuckle[i] = [lm.x, lm.y, lm.z]

                # 각 관절 사이의 벡터 크기(거리) 계산 
                v1 = knuckle[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19],:]  # Parent knuckle
                v2 = knuckle[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20],:]  # Child knuckle
                v = v2 - v1  # [20,3]
                
                # 벡터 크기를 1로 표준화(unit vector)
                v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

                # across(코사인의 역함수)에 대입하여 두 벡터가 이루는 각을 구함
                # 총 15개의 각도가 생성됨
                angle = np.arccos(np.einsum('nt,nt->n',
                    v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
                    v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:]))  # [15,]
                
                # radian에서 degree로 변환
                angle = np.degrees(angle)

                # 위에서 knn을 이용해 학습시킨 모델을 inference
                data = np.array([angle], dtype=np.float32)
                ret, results, neighbours, dist = knn.findNearest(data, 3)
                idx = int(results[0][0])

                # 손 제스처가 1이라면
                # 검지 손가락 위치에 circle을 그리고, 
                # index_finger_history에 검지 손가락의 위치와 색깔을 추가함
                if idx == 1:
                    x_pos = int(res.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x*frame_weight)
                    y_pos = int(res.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y*frame_height)
                    
                    cv.circle(frame,(x_pos, y_pos), 6, color, -1)
                    
                    index_finger_history.append([(x_pos, y_pos), color])
                    #index_finger_history.append((x_pos, y_pos))
                    
                # 그 외 제스처라면
                # 지금까지 그렸던 그림의 위치를 before_index_finger_history에 저장하고,
                # index_finger_history의 값들을 삭제함
                # 이를 통해, 손 이동 후에도 그림을 독립적으로 그릴 수 있음
                else:
                    before_index_finger_history.append(index_finger_history.copy())
                    #print(before_index_finger_history)
                    index_finger_history.clear()
        
        draw_line(frame, index_finger_history)
        
        for pos in before_index_finger_history:
            draw_line(frame, pos)

        if capture_save:
            cv.imwrite('finger draw' + str(save_picture_num) + '.jpg', frame)
            save_picture_num+=1
            capture_save = False
            
        cv.imshow('Draw by finger!', frame)
        cv.waitKey(1)

    cap.release()
    cv.destroyAllWindows()
    cv.waitKey(1)
    
    
app = QApplication(sys.argv)
pyqt_window = DrawByFinger()
pyqt_window.show()
app.exec_() 