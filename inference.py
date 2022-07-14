import os
import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import sys
from pathlib import Path


import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from config import parse_args
from recognition.model import Recognition_Model
from recognition.CTCConverter import CTCLabelConverter
from recognition.execute import *

SAVE_CWD = os.getcwd()
os.chdir(os.getcwd() + "/detection")
sys.path.append(os.getcwd())

from detection.models.common import DetectMultiBackend
from detection.utils.datasets import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from detection.utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from detection.utils.plots import Annotator, colors, save_one_box
from detection.utils.torch_utils import select_device, time_sync
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
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

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
    ######### Need to add font file

    ### Set up GPU
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    ### Read source
    source = str(args.source)
    save_img = not args.nosave and not source.endswith('.txt')  # save inference images


    ### check detection Directories
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
    (save_dir / 'labels' if args.save_bbox else save_dir).mkdir(parents=True, exist_ok=True)  # make dir


    ### Network Declare
    ## Detection
    detection_network = DetectMultiBackend(args.detect_weights, device=device, dnn=False, data=args.data, fp16=args.half)
    stride, names, pt = detection_network.stride, detection_network.names, detection_network.pt
    if len(args.detect_imgsz) ==1:
        args.detect_imgsz = [args.detect_imgsz[0], args.detect_imgsz[0]]
    imgsz = check_img_size(args.detect_imgsz, s=stride)  # check image size

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
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    batch_size = 1  # batch_size
    # vid_path, vid_writer = [None] * batch_size, [None] * batch_size
    detection_network.warmup(imgsz=(1 if pt else batch_size, 3, *imgsz))  # warmup

    time_line, seen = [0.0, 0.0, 0.0], 0

    ### start inference
    frame_idx = 0
    for path, im, im0s, vid_cap, s in dataset:
        # image preprocessing
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if detection_network.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        time_line[0] += t2 - t1

        # Do Detection Inference
        pred = detection_network(im, augment=False, visualize=False)
        t3 = time_sync()
        time_line[1] += t3 - t2

        # Detection NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, None, False, max_det=args.max_det)
        time_line[2] += time_sync() - t3


        # Process predictions
        for i, det in enumerate(pred):  # per predictions
            seen += 1
            p, img, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            # Rescale boxes from img_size to img size
            t3 = time_sync()
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img.shape).round()
            time_line[2] += time_sync() - t3

            # pred results
            bbox = det[:, :4]   # [pred_num, 4]    [x1, y1, x2, y2]    ([] if failed to pred, not normalized)
            conf = det[:, 4:5]  # [pred_num, 1]                        ([] if failed to pred)
            # img = img         # [H, W, 3],      0 ~ 255 normalized

            # saving detection result image and bbox.txt -> 차후에 recognition이랑 통합?
            s = save_detection_result(args, save_img, save_dir, det, names, s, p, dataset.mode, frame, img.copy(), im)


        #########################
        ###### Recognition ######
        #########################
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rec = toTensor(img).to(device)
        recog_result = do_recognition(args, img_rec, bbox, recognition_network, converter, device)
        img = draw_result(img, bbox, recog_result, font, frame_idx, args.save_dir)

        # Print detection time
        if args.print_detect:
            LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')

        frame_idx += 1

    img2video(args.save_dir, args.save_videoname)

    # Print detection results
    t = tuple(x / seen * 1E3 for x in time_line)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
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


def save_detection_result(args, save_img, save_dir, det, names, s, p,mode, frame, imc, im):
    # prepare detection result path
    p = Path(p)  # to Path
    save_path = str(save_dir / p.name)  # im.jpg
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if mode == 'image' else f'_{frame}')  # im.txt
    s += '%gx%g ' % im.shape[2:]  # print string
    gn = torch.tensor(imc.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(imc, line_width=3, example=str(names))

    # prepare Print results
    for c in det[:, -1].unique():
        n = (det[:, -1] == c).sum()  # detections per class
        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

    for *xyxy, conf, cls in reversed(det):
        if args.save_bbox:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        if save_img or args.save_crop:  # Add bbox to image
            c = int(cls)  # integer class
            label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
            if args.save_crop:
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    # Stream results
    im0 = annotator.result()

    # Save results (image with detections)
    if save_img:
        if mode == 'image':
            cv2.imwrite(save_path, im0)
        elif mode == 'video':
            im_save_path = save_path[:-4] + "_" + str(frame) + ".png"
            cv2.imwrite(im_save_path, im0)
        """
        # video로 save하는 코드 -> recognition 까지 완성한 후 delete? or 추가?
        else:  # 'video'
            if vid_path[i] != save_path:  # new video
                vid_path[i] = save_path
                if isinstance(vid_writer[i], cv2.VideoWriter):
                    vid_writer[i].release()  # release previous video writer
                if vid_cap:  # video
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                else:  # stream
                    fps, w, h = 30, im0.shape[1], im0.shape[0]
                save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            vid_writer[i].write(im0)
        """

    return s



if __name__ == "__main__":
    main()
