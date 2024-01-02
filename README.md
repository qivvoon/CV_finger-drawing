# Draw by finger 🎨
노트북 카메라를 통해 출력되는 실시간 영상에 그림을 그리는 프로그램이다. </br>
노트북 화면을 터치하여 그리는 것이 아닌 손동작을 이용하여 허공에 그림을 그리는 것이 특징이다. <br></br>

## 기술 스택
<img src="https://img.shields.io/badge/Mediapipe-1A86FD"/> <img src="https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white"/> <img src="https://img.shields.io/badge/Anaconda-44A833?style=flat&logo=anaconda&logoColor=white"/> <img src="https://img.shields.io/badge/Spyder-FF0000?style=flat&logo=spyderide&logoColor=white"/> <img src="https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white"/> <img src="https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white"/> <img src="https://img.shields.io/badge/PyQt5-3C2179"/>

## 주요 프로그래밍
손동작 판별을 위해 `Mediapipe`와 `KNN알고리즘`을 사용하였다.
### 로직
KNN 모델 학습 -> 학습 된 KNN을 이용하여 프로그램이 손동작 판별
1. KNN 학습 </br>
   -
   * OpenCV에서 제공하는 KNN 알고리즘 이용
   <img width="1292" alt="스크린샷 2024-01-02 오후 4 42 28" src="https://github.com/qivvoon/CV_finger-drawing/assets/90748096/9797b2e1-2fab-471c-84dc-230645076483">

2. 학습된 KNN 이용
   -
    <img width="1292" alt="스크린샷 2024-01-02 오후 4 48 10" src="https://github.com/qivvoon/CV_finger-drawing/assets/90748096/6519cc7a-50ce-4f63-8c5e-d71bffe8fe0b">

## 사용법 및 기능
