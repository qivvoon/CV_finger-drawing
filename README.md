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
   <img width="1100" alt="스크린샷 2024-01-02 오후 4 42 28" src="https://github.com/qivvoon/CV_finger-drawing/assets/90748096/9797b2e1-2fab-471c-84dc-230645076483">

2. 학습된 KNN 이용
   -
    <img width="1100" alt="스크린샷 2024-01-02 오후 4 48 10" src="https://github.com/qivvoon/CV_finger-drawing/assets/90748096/6519cc7a-50ce-4f63-8c5e-d71bffe8fe0b">
<br></br>
## 사용법 및 기능
프로그램 실행 시, 아래와 같은 화면이 나타난다. <br></br>
<img width="403" alt="스크린샷 2024-01-02 오후 4 56 39" src="https://github.com/qivvoon/CV_finger-drawing/assets/90748096/1b2c670a-6354-472b-a8aa-9cb8bd7e06e1">

* `Start Draw`
  - 카메라가 켜지고, 실시간 영상이 출력된다.
  - 아래 사진과 같은 손동작을 취하면 영상에 그림을 그릴 수 있다. 그 외 손동작(주먹, 보 등)으로는 그림을 그릴 수 없다. </br>
    <img width="150" alt="image" src="https://github.com/qivvoon/CV_finger-drawing/assets/90748096/34c4f55a-52ad-400a-9b2f-c0e1ea8a3f57">

* `Change Color`
  - `Change Color` 버튼 밑의 `Combo box`에서 색상을 선택한 후 `Change Color`를 클릭하면, 선택된 색상으로 그림을 그릴 수 있다.
  - 색상은 검정색, 빨간색, 노란색이 있으며, 기본 값은 검정색이다.
 
* `Save`
  - `Save`를 클릭한 순간의 영상을 사진으로 저장한다.
  - Draw by finger가 설치되어 있는 폴더에 저장된다.
  - 사진 파일의 이름은 `finger draw숫자` 형태로 저장된다. 처음 저장되는 사진의 숫자는 0이며, 저장될 때마다 1씩 증가한다. 예) `finger draw0`

 * `Erase`
   - 지금까지 그렸던 그림을 전부 삭제한다.

* `Exit`
  - 프로그램을 종료한다.
