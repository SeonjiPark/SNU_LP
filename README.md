# SNU_LP

# Environments
Pytorch >= 1.7.0

Python >= 3.7.0

git clone https://github.com/SeonjiPark/SNU_LP.git

cd SNU_LP

conda create -n SNU_LP python=3.7

conda activate SNU_LP

pip install -r requirements.txt

# Dataset
Z 서버에서 SHARE안에 teset dataset 있음

# Directory 설명
    |── datasets
                ├──> train
                ├──> images : 학습 이미지가 저장되야 하는 폴더
                └──> label.txt : 적절한 입력 포멧으로 변형한 학습 레이블
            ├──> val
                ├──> images : 검증 이미지가 저장되야 하는 폴더
                └──> label.txt : 적절한 입력 포멧으로 변형한 검증 레이블
    |── SNU_LP
        ├──> detections : detection용 train, test, inference.py
        ├──> inference_result (inference.py 실행시 생김)
            ├──> recognition 
            ├──> detection (save_bbox or save_detect_img를 True로 줄시 생김)
                    └──> labels : 이미지에 대한 실행 결과 (bbox, confidence를 저장)
                ├── 실행 결과 이미지 (bbox, confidence 포함)
        ├──> models : detection yolov5 layers
        ├──> weights
            ├──> AD_15_E300_scratch_anchor_ver1 : 중복파일 (삭제예정)
            ├── best.pt : 현재 best ckpt
            ├── yolov5s.pt : yolov5 pretrained ckpt (삭제예정?)

        |── utils : 다양한 기타 사용 함수들 폴더
        |── config.py : 입력 argument를 관리하는 파일
        |── inference.py : inference용 코드 (GT label이 없을 경우 테스트)
        |── requirements.txt : 가상환경 파일


# 코드 실행 가이드 라인


## === Train ===
추가 예정

## === Test ===
추가 예정

## === Inference ===
python inference.py --source ./road_driving.mp4 --device 0

inference 실행 시 디폴트는 이미지 저장 O, txt는 저장 X
위와 같이 --save-bbox 를 추가하면 bbox를 txt로 저장함. 


code 내부에서 return 하는 것 

image : [H, W, C] 원본 이미지. 사이즈는 원본 사이즈 그대로, 0~255 normalize 

bboxes : [pred_num, 4] 해당 이미지에서 predict한 bbox. normalized 하지 않은 [x1, y1, x2, y2]


### 주의 : bbox txt 파일에는 normalize된 center_x, center_y, w, h가 저장됨. (return 값과 다름)


알아둬야 할 config 설명

--detect_weights weights/best.pt 로 고정

--source 인풋 이미지 폴더명 or 파일명 (동영상 가능)

--save_bbox : 플래그로 줄 시 runs 안에 class와 bbox를 txt로 저장

--save_conf : 플래그로 줄 시 runs 안에 bbox.txt에 conf 추가

--save_detect_img : 플래그로 줄 시 detection 이미지 저장 O 

--conf_thres : float로 주기 가능. default는 0.9 
