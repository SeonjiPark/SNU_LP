# SNU_LP

# Environments
Pytorch >= 1.7.0

Python >= 3.7.0

git clone https://github.com/SeonjiPark/SNU_LP.git

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
    └──> inference.py : inference 용 코드 (GT label이 없는 경우)

## === 학습된 ckpt ===
아래 링크에서 미리 학습된 ckpt 파일을 다운 받아 weights 폴더에 배치

구글 드라이브 주소 : https://drive.google.com/drive/folders/112Lt3OqficYWn61HwqbJQmm7DIkGPkfA?usp=sharing

## === Inference ===
```
python inference.py --gpu_num=0 --source='test.mp4' --save_result_image=True --save_result_video=True --save_dir='inference_result/' --save_videoname='out.mp4'
```

Argument 설명

--source : 인풋 이미지 폴더명 or 파일명 (동영상 가능)

--save_result_image : detection & recognition 결과 이미지로 저장할지 여부

--save_result_video : detection & recognition 결과 동영상으로 저장할지 여부 (save_result_image=True 인 경우만 가능)

--save_dir : Inference 결과 저장할 폴더

-save_videoname : 저장할 Video 제목



[detection 결과 저장 관련 arg]

--save_detect_img: detection 결과 이미지를 저장할지 여부 (save_result_image=True일때만 가능, default=True)

--save_bbox: detection 결과 bbox를 txt로 저장할지 여부 (save_result_image=True일때만 가능, default=False)

#### 주의 : bbox txt 파일에는 normalize된 center_x, center_y, w, h가 저장됨. (return 값과 다름)

 

## === Code 내부에서 return 하는 것 ===
code 내부에서 return 하는 것 

image : [H, W, C] 원본 이미지. 사이즈는 원본 사이즈 그대로, 0~255 normalize 

bboxes : [pred_num, 4] 해당 이미지에서 predict한 bbox. normalized 하지 않은 [x1, y1, x2, y2]

