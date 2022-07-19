import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
from pathlib import Path
from time import time

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from config import parse_args
from recognition.model import Recognition_Model
from recognition.CTCConverter import CTCLabelConverter
from recognition.execute import *

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

from detection.utils.datasets import LoadImages
from detection.utils.general import colorstr, increment_path
from detection.execute import preprocess_img, do_detect, save_detection_result, build_detect_model
os.chdir(SAVE_CWD)

province = ['대구서', '동대문', '미추홀', '서대문', '영등포', '인천서', '인천중',
                    '강남', '강서', '강원', '경기', '경남', '경북', '계양', '고양', '관악', '광명', '광주', '구로', '금천', '김포', '남동', 
                    '대구', '대전', '동작', '부천', '부평', '서울', '서초', '안산', '안양', '양천', '연수', '용산', '인천', '전남', '전북', 
                    '충남', '충북', '영']

province_replace = ['괅', '놝', '돩', '랅', '맑', '밝', '삵', '앍', '잙', '찱',
                    '괉', '놡', '돭', '랉', '맕', '밡', '삹', '앑', '잝', '찵',
                    '괋', '놣', '돯', '뢇', '맗', '밣', '삻', '앓', '잟', '찷',
                    '괇', '놟', '돫', '뢃', '맓', '밟', '삷', '앏', '잛', '찳']

chars = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '가', '거', '고', '구', '나', '너', '노', '누', '다', '더', '도', '두', 
        '라', '러', '로', '루', '마', '머', '모', '무', '바', '배', '버', '보', '부', '사', '서', '소', '수', '시', '아', '어', '오', 
        '우', '육', '자', '저', '조', '주', '지', '차', '카', '타', '파', '하', '허', '호', '히']

chars = chars + province_replace

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

def main():
    args = parse_args()

    ### Print Arguments
    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))

    ### Converter for Recognition network
    args.input_channel = 3
    converter = CTCLabelConverter(chars)
    args.num_class = len(converter.character)

    ### Font for drawing Korean
    fontpath = "gulim.ttc"
    font = ImageFont.truetype(fontpath, 20)
    toTensor = transforms.ToTensor()

    ### Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Network Declare
    ## Detection
    detection_network, imgsz, stride, names, pt = build_detect_model(args, device)

    ## Recognition
    recognition_network = Recognition_Model(args, device)
    recognition_network.to(device)
    # Load network weight
    recognition_checkpoint = torch.load(args.weight_dir + args.recognition_weight)
    recognition_network.load_state_dict(recognition_checkpoint['network'])
    with torch.no_grad():
        detection_network.eval()
        recognition_network.eval()

    print("Load Network Weights Done!")

    ### Make Save Directories
    os.makedirs(args.save_dir, exist_ok=True)
    exp_num = len(os.listdir(args.save_dir))
    EXP_NAME = args.save_dir + 'exp' + str(exp_num).zfill(4)
    os.makedirs(EXP_NAME, exist_ok=True)
    if args.save_result_image:
        RECOG_SAVE_DIR = EXP_NAME + '/recognition/'
        DETECT_SAVE_DIR = EXP_NAME + '/detection/'
        os.makedirs(RECOG_SAVE_DIR, exist_ok=True)
        os.makedirs(DETECT_SAVE_DIR, exist_ok=True)
        os.makedirs(DETECT_SAVE_DIR + '/labels' if args.save_bbox else DETECT_SAVE_DIR, exist_ok=True)
    else:
        RECOG_SAVE_DIR = None
        DETECT_SAVE_DIR = None

    ### Load Datas
    ## Read source
    source = str(args.source)
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)

    ### start inference
    frame_idx = 0
    time_line = [0.0, 0.0, 0.0, 0.0, 0.0] # GPU_load, do_detect, do_recog, draw_result, img2video
    for path, img0, cap, s, img_size, stride, auto in dataset:
        # img [H, W, C], 0~255 normalized
        start = time()
        img = torch.from_numpy(img0).to(device)
        time_line[0] += time() - start

        # Do Detection Inference
        start = time()
        preds = do_detect(args, detection_network, img, img_size, stride, auto)
        time_line[1] += time() - start
        bboxes = preds[:, :4]  # bbox = [pred_num, 4]    [x1, y1, x2, y2]    ([] if failed to pred, not normalized)

        img = img[:, :, [2, 1, 0]]  # RGB -> BGR
        img_rec = img.permute(2, 0, 1) / 255 # HWC -> CHW

        start = time()
        recog_result = do_recognition(args, img_rec, bboxes, recognition_network, converter, device)
        time_line[2] += time() - start

        if args.save_result_image:
            start = time()
            draw_result(img, preds, recog_result, font, frame_idx, RECOG_SAVE_DIR,
                        dataset, args, names, path, DETECT_SAVE_DIR)
            time_line[3] += time() - start

        frame_idx += 1

    if args.save_result_image and args.save_result_video:
        start = time()
        img2video(EXP_NAME, RECOG_SAVE_DIR, args.save_videoname, dataset.fps)
        time_line[4] += time() - start

    print(f'Inference on {frame_idx} frames has done !')
    print(f'Total executing time is {sum(time_line[:5]):.3f} s = Inferencing ({sum(time_line[:3]):.3f}s) + Saving Result ({sum(time_line[3:]):.3f})s')
    print(f'Inference time per frame is {sum(time_line[:3]) / frame_idx:.3f}s = (GPU load {time_line[0]/frame_idx:.3f}s) + (Detection {time_line[1]/frame_idx:.3f}s) + (Recognition {time_line[2]/frame_idx:.3f}s)')
    print(f'Saving image results takes {time_line[3] / frame_idx:.3f}s per frame, and saving video result takes {time_line[4]:.3f}s')

def draw_result(img, preds, recog_result, font, frame_idx, RECOG_SAVE_DIR, dataset, args, names, path, DETECT_SAVE_DIR):
    img = np.asarray(img.to("cpu"))
    bboxes = preds[:, :4]

    if args.save_bbox or args.save_detect_img:
        frame = getattr(dataset, 'frame', 0)
        save_detection_result(args, preds, names, path, dataset.mode, frame, img.copy(), DETECT_SAVE_DIR)

    bbox_num = len(bboxes)

    for idx in range(bbox_num):  
        cur_lp = bboxes[idx].detach().cpu().numpy()
        # Draw bbox
        cv2.rectangle(img, (int(cur_lp[0]), int(cur_lp[1])), (int(cur_lp[2]), int(cur_lp[3])), color=(0,0,255), thickness=8)

        # Draw recognition result    
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)

        decode_pred = decode_province(recog_result[idx], province, province_replace)

        draw.text((int(cur_lp[0])-40, int(cur_lp[1])-40), decode_pred, font=font, fill=(0,0,255,0))

        img = np.array(img_pil)

    outname = RECOG_SAVE_DIR + str(frame_idx+1).zfill(3) + '.jpg'
    cv2.imwrite(outname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def img2video(save_dir, frame_dir, video_name, fps):

    if '.mp4' in video_name:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    else:
        fourcc = 0

    frames = [frame for frame in os.listdir(frame_dir) if frame.endswith(".jpg")]
    frames = sorted(frames)
    frame = cv2.imread(os.path.join(frame_dir, frames[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(os.path.join(save_dir, video_name), fourcc, fps, (width, height))
    
    for frame in frames:
        video.write(cv2.imread(os.path.join(frame_dir, frame)))
    
    video.release()


if __name__ == "__main__":
    main()
