# SNU_LP

# Environments
Pytorch >= 1.7.0

Python >= 3.7.0

git clone -b API --single-branch https://github.com/SeonjiPark/SNU_LP.git

cd SNU_LP

```
conda create -n ENV_NAME python=3.7
conda activate ENV_NAME
pip install -r requirements.txt
```

# Directory 설명
    |── detection : detection 관련 코드
    |── recognition : recognition 관련 코드
    |── weights : pretrained detection & recognition weight들 저장
    |── config.py : 입력 arugment 를 관리하는 파일
    |── gulim.ttc : 한글 출력을 위한 폰트
    └──> detect.py : inference 용 코드 (GT label이 없는 경우)

## === 학습된 ckpt ===
아래 링크에서 미리 학습된 ckpt 파일을 다운 받아 weights 폴더에 배치

구글 드라이브 주소 : https://drive.google.com/drive/folders/112Lt3OqficYWn61HwqbJQmm7DIkGPkfA?usp=sharing

## === Inference ===
```
python detect.py
```

Argument (detect.cfg) 설명



[detection 결과 저장 관련 arg]


#### 주의 : bbox txt 파일에는 normalize된 center_x, center_y, w, h가 저장됨. (return 값과 다름)

 

## === Code 내부에서 return 하는 것 ===


