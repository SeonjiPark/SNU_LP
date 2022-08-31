import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
from pathlib import Path
from time import time
import configparser
import argparse
import copy

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

from recognition.model import Recognition_Model
from recognition.CTCConverter import CTCLabelConverter
from recognition.execute import *

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

# from detection.utils.datasets import LoadImages
from detection.utils.general import colorstr, increment_path
from detection.execute import LoadDatas, do_detect, save_detection_result, build_detect_model

from detection.utils.plots import Annotator, colors
from detection.utils.general import xyxy2xywh

"""
pip install PyYAML>=5.3.1
pip install pandas>=1.1.4
pip install matplotlib>=3.2.2
pip install seaborn>=0.11.0
"""
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

IMAGE_FORMATS = '.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp'  
VIDEO_FORMATS = '.asf', '.avi', '.gif', '.m4v', '.mkv', '.mov', '.mp4', '.mpeg', '.mpg', '.ts', '.wmv' 

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

def init_snu(config_file):

    ################################################
    #   1. Read in parameters from config file     #
    ################################################
    parser = argparse.ArgumentParser()

    args = parser.parse_args()

    config = configparser.RawConfigParser()
    config.read(config_file)

    basic_config = config["basic_config"]

    # Weight Files
    args.detection_weight_file = basic_config["detection_weight_file"]
    if not os.path.exists(args.detection_weight_file):
        print(">>> NOT Exist DETECTION WEIGHT File {0}".format(args.detection_weight_file))
        #sys.exit(2)

    args.recognition_weight_file = basic_config["recognition_weight_file"]
    if not os.path.exists(args.recognition_weight_file):
        print(">>> NOT Exist RECOGNITION WEIGHT File {0}".format(args.recognition_weight_file))
        #sys.exit(2)

    # Input Data File
    args.source = basic_config["source"]
    if not os.path.exists(args.source):
        print(">>> NOT Exist INPUT File {0}".format(args.source))

    # GPU Number
    args.gpu_num = basic_config["gpu_num"]
    if args.gpu_num == "" :
        print(">>> NOT Assign GPU Number")
        #sys.exit(2)

    # Detection Parameters
    args.infer_imsize_same = basic_config.getboolean('infer_imsize_same')
    if args.infer_imsize_same == "" :
        args.infer_imsize_same = False

    args.detect_save_library = basic_config.getboolean('detect_save_library')
    if args.detect_save_library == "" :
        args.detect_save_library = False

    args.data = basic_config["data"]
    if args.data == "" :
        args.data = 'detection/data/AD.yaml'

    args.half = basic_config.getboolean('half')
    if args.half == "" :
        args.half = False

    imgsz = int(basic_config["detect_imgsz"])
    args.detect_imgsz = [imgsz]
    if args.detect_imgsz == "" :
        args.detect_imgsz = [640]

    args.conf_thres = float(basic_config["conf_thres"])
    if args.conf_thres == "" :
        args.conf_thres = 0.9

    args.iou_thres = float(basic_config["iou_thres"])
    if args.iou_thres == "" :
        args.iou_thres = 0.45

    args.max_det = int(basic_config["max_det"])
    if args.max_det == "" :
        args.max_det = 1000


    # Recognition Parameters
    args.Transformation = basic_config["transformation"]
    if args.Transformation == "" :
        args.Transformation = 'TPS'

    args.FeatureExtraction = basic_config["featureExtraction"]
    if args.FeatureExtraction == "" :
        args.FeatureExtraction = 'ResNet'

    args.SequenceModeling = basic_config["sequenceModeling"]
    if args.SequenceModeling == "" :
        args.SequenceModeling = 'BiLSTM'

    args.Prediction = basic_config["prediction"]
    if args.Prediction == "" :
        args.Prediction = 'CTC'

    args.num_fiducial = int(basic_config["num_fiducial"])
    if args.num_fiducial == "" :
        args.num_fiducial = 20

    args.imgH = int(basic_config["imgH"])
    if args.imgH == "" :
        args.imgH = 64 

    args.imgW = int(basic_config["imgW"])
    if args.imgW == "" :
        args.imgW = 200

    args.output_channel = int(basic_config["output_channel"])
    if args.output_channel == "" :
        args.output_channel = 512

    args.hidden_size = int(basic_config["hidden_size"])
    if args.hidden_size == "" :
        args.hidden_size = 256

    args.input_channel = 3

    ### Print Arguments
    for key,value in sorted((args.__dict__).items()):
        print('\t%15s:\t%s' % (key, value))


    # Add Directory & Result save parameters
    args.output_dir = basic_config["output_dir"]
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.result_savefile = basic_config.getboolean('result_savefile')
    if args.result_savefile == "" :
        args.result_savefile = False

    args.save_detect_result = basic_config.getboolean('save_detect_result')
    if args.save_detect_result == "" :
        args.save_detect_result = False

    args.save_recog_result = basic_config.getboolean('save_recog_result')
    if args.save_recog_result == "" :
        args.save_recog_result = False

    args.hide_labels = basic_config.getboolean('hide_labels')
    if args.hide_labels == "" :
        args.hide_labels = False

    args.hide_conf = basic_config.getboolean('hide_conf')
    if args.hide_conf == "" :
        args.hide_conf = False

    args.save_conf = basic_config.getboolean('save_conf')
    if args.save_conf == "" :
        args.save_conf = False


    # Other parameters
    args.deidentified_type = basic_config["deidentified_type"]
    if args.deidentified_type == "" :
        args.deidentified_type = 2

    args.recognition_library_path = basic_config["recognition_library_path"]
    if not os.path.exists(args.recognition_library_path):
        print(">>> NOT Exist RECOGNITION LIBRARY {0}".format(args.recognition_library_path))


    ################################################
    #        1.5 Converter for Recognition         #
    ################################################
    converter = CTCLabelConverter(chars)
    args.num_class = len(converter.character)

    ################################################
    #                2. Set up GPU                 #
    ################################################
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ################################################
    #  3. Declare Detection / Recognition Network  #
    ################################################
    # Detection Network
    args.detect_weights = args.detection_weight_file
    detection_network, imgsz, stride, names, pt = build_detect_model(args, device)
    # Add network parameters to args
    args.pt = pt
    args.stride = stride
    args.imgsz = imgsz
    args.names = names

    # Recognition Network
    recognition_network = Recognition_Model(args, device)
    recognition_network.to(device)

    ################################################
    #    4. Load Detection / Recognition Network   #
    ################################################
    recognition_checkpoint = torch.load(args.recognition_weight_file, map_location=device)
    recognition_network.load_state_dict(recognition_checkpoint['network'])
    with torch.no_grad():
        detection_network.eval()
        recognition_network.eval()    

    ################################################
    #      5. Make Result Saving Directories       #
    ################################################
    os.makedirs(args.output_dir, exist_ok=True)

    return (args, device, converter, detection_network, recognition_network)

def read_data_snu(source):
    ### Load Datas
    source = str(source)
    dataset = LoadDatas(source)
    return dataset

def detect_snu(args, device, dataset, detection_network):
    ### start inference
    frame_idx = 0
    # time_line = [0.0, 0.0, 0.0, 0.0, 0.0]  # GPU_load, do_detect, do_recog, draw_result, img2video
    detect_preds = []
    original_images = []

    for path, img in dataset:
        # img [H, W, C], 0~255 normalized
        img = torch.from_numpy(img).to(device)

        # Do Detection Inference
        preds = do_detect(args, detection_network, img, args.imgsz, args.stride, auto=True)
        # bboxes = preds[:, :4]  # bbox = [pred_num, 4]    [x1, y1, x2, y2]    ([] if failed to pred, not normalized)
        # preds = preds.detach().cpu().numpy()
        detect_preds.append(preds)

        img = img[:, :, [2, 1, 0]]  # RGB -> BGR
        img_rec = img.permute(2, 0, 1) / 255  # HWC -> CHW
        original_images.append(img_rec)

        frame_idx += 1
    return original_images, detect_preds

def recognize_snu(args, detect_preds, device, converter, recognition_network, original_images):

    ### Start Recognition
    recog_preds = []

    for (img, bboxes) in zip(original_images, detect_preds):
    
        # Recognition Result : Not decoded / Provinces are in the format of '괅','놝', etc...
        recog_result = do_recognition(args, img, bboxes, recognition_network, converter, device)

        bbox_num = len(bboxes)

        cur_recog_result = []

        # Recognition result to numpy list (x1, y1, x2, y2, decoded characeter)
        for bbox_idx in range(bbox_num):

            cur_bbox = bboxes[bbox_idx]
            cur_lp = cur_bbox.detach().cpu().numpy()

            x1, y1, x2, y2, conf, cls = int(cur_lp[0]), int(cur_lp[1]), int(cur_lp[2]), int(cur_lp[3]), cur_lp[4], cur_lp[5]
            decode_pred = decode_province(recog_result[bbox_idx], province, province_replace)

            append_row = [x1, y1, x2, y2, conf, cls, decode_pred]
            cur_recog_result.append(append_row)

        recog_preds.append(cur_recog_result)

    return recog_preds
            
def save_result_snu(args, images, recog_preds):
    INPUT_IMAGE_FILE_ONLYNM = os.path.basename(str(args.source))
    file_ext = os.path.splitext(INPUT_IMAGE_FILE_ONLYNM)[-1]
    file_nm = os.path.splitext(INPUT_IMAGE_FILE_ONLYNM)[0]
    OUTPUT_IMAGE_FILE = os.path.join(args.output_dir, file_nm + "_recognized" + file_ext)

    # In case input is folder (not image or video)
    if file_ext == "":
        file_nm = file_nm.split("/")[-1]

    OUTPUT_BASE_DIR = os.path.join(args.output_dir, file_nm)
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    args.output_base_dir = OUTPUT_BASE_DIR

    if args.result_savefile:
        RECOG_SAVE_DIR = os.path.join(args.output_dir, file_nm, "recognition")
        DETECT_SAVE_DIR = os.path.join(args.output_dir, file_nm, "detection")
        LABEL_SAVE_DIR = os.path.join(args.output_dir, file_nm, "labels")

        os.makedirs(RECOG_SAVE_DIR, exist_ok=True)
        os.makedirs(DETECT_SAVE_DIR, exist_ok=True)
        os.makedirs(LABEL_SAVE_DIR, exist_ok=True)
    else:
        RECOG_SAVE_DIR = None
        DETECT_SAVE_DIR = None
        LABEL_SAVE_DIR = None

    args.recog_save_dir = RECOG_SAVE_DIR
    args.detect_save_dir = DETECT_SAVE_DIR
    args.label_save_dir = LABEL_SAVE_DIR

    if file_ext in VIDEO_FORMATS:
        args.isvideo = True
    else:
        args.isvideo = False

    print("{0}\n{1}\n{2}\n".format(INPUT_IMAGE_FILE_ONLYNM, file_ext, file_nm))
    args.input_onlynm = INPUT_IMAGE_FILE_ONLYNM
    args.input_ext = file_ext
    args.input_filenm = file_nm

    if args.result_savefile:
        for idx, img in enumerate(images):
            ## save detect result
            img = img.permute(1,2,0) * 255
            img = img[:, :, [2, 1, 0]]
            img = img.cpu().numpy().astype(np.uint8)
            
            if args.save_detect_result:
                annotator_detect = Annotator(img, line_width=3, pil=True, example=str(args.names))
            if args.save_recog_result:
                annotator_recog = Annotator(img, line_width=3, font='gulim.ttc', font_size=40, pil=True, example=str(args.names))
            gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

            savefile = args.input_filenm + "_detection_" + str(idx + 1).zfill(10)

            for *xyxy, conf, cls, lp_characters in reversed(recog_preds[idx]):
                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format

                # Save label
                TXT_PATH = os.path.join(LABEL_SAVE_DIR, savefile + ".txt")
                with open(TXT_PATH, 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                c = int(cls)  # integer class

                # Save Detection Result
                if args.save_detect_result:
                    if args.hide_labels and args.hide_conf:
                        label = None
                    elif args.hide_labels and not args.hide_conf:
                        label = f'{conf:.2f}'
                    elif not args.hide_labels and args.hide_conf:
                        label = args.names[c]
                    else:
                        label = f'{args.names[c]} {conf:.2f}'

                    annotator_detect.box_label(xyxy, label, color=colors(c, True))

                # Save Recognition Result
                if args.save_recog_result:
                    label = f'{lp_characters}'
                    annotator_recog.box_label(xyxy, label, color=colors(c,True))

            # Stream results
            if args.save_detect_result:
                im_detect = annotator_detect.result()
                outname = os.path.join(DETECT_SAVE_DIR, savefile + '.jpg')
                cv2.imwrite(outname, im_detect)

            if args.save_recog_result:
                im_recog = annotator_recog.result()
                outname = os.path.join(RECOG_SAVE_DIR, savefile + '.jpg')
                cv2.imwrite(outname, im_recog)


if __name__ == "__main__":
    """
     python detect.py --source='test_resources/near.jpg' --output_dir='inference_result/'
    """
    (args, device, converter, detection_network, recognition_network) = init_snu('detect.cfg')

    print("Read Dataset")
    dataset = read_data_snu(args.source)

    print("Run Detection")
    original_images, detect_preds = detect_snu(args, device, dataset, detection_network)

    print("Run Recognition.")
    recog_preds = recognize_snu(args, detect_preds, device, converter, recognition_network, original_images)

    print("Save Result")
    save_result_snu(args, original_images, recog_preds)

    print("Done!")
