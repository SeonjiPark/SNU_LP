import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
from pathlib import Path

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
from detection.utils.general import LOGGER, colorstr, cv2, increment_path, xyxy2xywh
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

    ### check detection Directories
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
    (save_dir / 'labels' if args.save_bbox else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

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


    ### Load Datas
    ## Read source
    source = str(args.source)
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)


    ### start inference
    frame_idx = 0
    for path, img0, cap, s, img_size, stride, auto in dataset:
        # image preprocessing
        img = torch.from_numpy(img0).to(device)
        im_resize = preprocess_img(img, detection_network.fp16, img_size, stride, auto)

        # Do Detection Inference
        pred = do_detect(args, detection_network, im_resize, img0.copy())
        bbox = pred[:, :4]  # [pred_num, 4]    [x1, y1, x2, y2]    ([] if failed to pred, not normalized)
        # img [H, W, C], 0~255 normalized

        #########################
        ###### Recognition ######
        #########################
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img_rec = toTensor(img).to(device)
        # recog_result = do_recognition(args, img_rec, bbox, recognition_network, converter, device)
        # img = draw_result(img, bbox, recog_result, font, frame_idx, args.save_dir)

        # saving detection result image and bbox.txt -> 차후에 recognition이랑 통합?
        save_detection_result(args, save_img, save_dir, pred, names, path, dataset.mode, getattr(dataset, 'frame', 0), np.asarray(img.to("cpu")))

        # Print detection time
        frame_idx += 1

    img2video(args.save_dir, args.save_videoname)

    # Print detection results
    if args.save_bbox or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_bbox else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")


def draw_result(img, bboxes, recog_result, font, frame_idx, SAVE_DIR):

    os.makedirs(SAVE_DIR, exist_ok=True)

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

    outname = SAVE_DIR + str(frame_idx).zfill(3) + '.jpg'
    cv2.imwrite(outname, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def img2video(frame_dir, video_name):

    if '.mp4' in video_name:
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    else:
        fourcc = 0

    frames = [frame for frame in os.listdir(frame_dir) if frame.endswith(".jpg")]
    frames = sorted(frames)
    frame = cv2.imread(os.path.join(frame_dir, frames[0]))
    height, width, layers = frame.shape

    video = cv2.VideoWriter(video_name, fourcc, 24.0, (width, height))
    
    for frame in frames:
        video.write(cv2.imread(os.path.join(frame_dir, frame)))
    
    video.release()


if __name__ == "__main__":
    main()
