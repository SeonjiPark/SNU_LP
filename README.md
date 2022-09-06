# SNU_LP

# Dataset Directory 설명

1. CCPD (중국 번호판)
    |── CCPD2019 |── ccpd_base - images
                 |── ccpd_blur
                 |── ccpd_challenge
                 |── ccpd_db
                 |── ccpd_fn
                 |── ccpd_np
                 |── ccpd_rotate
                 |── ccpd_tilt
                 |── ccpd_weather
                 └──> splits - ccpd_blur.txt, ccpd_challenge.txt, test.txt, val.txt, train.txt ....

2. KorLP (한국 번호판)
    |── KorLP |── Training    |── image - images
                              └──> label -labels
              └──> Validation |── image - images
                              └──> label -labels



# Directory 설명
    |── experiments : weight 저장하는 폴더
    |── modules : recognition에 필요한 module들 (model.py 에서 사용)
    |── utils : converter / data_loader / preprocess 등
    |── config.py : 입력 arugment 를 관리하는 파일
    |── evaluate.py : evaluation 코드
    |── gulim.ttc : 한글 출력을 위한 폰트
    |── kor_char_information.py : 한국 번호판 Character 정보
    |── model.py : Recognition model
    |── train_ccpd.py : 중국 번호판 학습
    └──> train_kor.py : 한국 번호판 학습



## === 학습된 ckpt ===
1. 아래 링크에서 미리 학습된 ckpt 파일을 다운
2. recognition.pth -> ckpt_best.pth 로 rename
3. experiment - "experiment_name" - ckpt - 밑에 locate하기

구글 드라이브 주소 : https://drive.google.com/drive/folders/112Lt3OqficYWn61HwqbJQmm7DIkGPkfA?usp=sharing



# 학습 시 주의할 사항
config.py 에서

--batch_max_length : Korean 할 때는 9 / Chinese 할 때는 7


