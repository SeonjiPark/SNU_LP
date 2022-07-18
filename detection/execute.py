import cv2
import torch
import numpy as np
from pathlib import Path

from utils.general import  non_max_suppression, scale_coords, check_img_size, increment_path
from utils.augmentations import letterbox
from utils.plots import Annotator, colors, save_one_box

from models.common import DetectMultiBackend

def build_detect_model(args, device):
    detection_network = DetectMultiBackend(args.detect_weights, device=device, dnn=False, data=args.data, fp16=args.half)
    stride, names, pt = detection_network.stride, detection_network.names, detection_network.pt
    if len(args.detect_imgsz) ==1:
        args.detect_imgsz = [args.detect_imgsz[0], args.detect_imgsz[0]]
    imgsz = check_img_size(args.detect_imgsz, s=stride)  # check image size
    detection_network.warmup(imgsz=(1, 3, *imgsz))  # warmup

    return detection_network, imgsz, stride, names, pt


def preprocess_img(img, fp_flag, img_size, stride, auto):
    img = letterbox(img, img_size, stride=stride, auto=auto)[0]
    img = img.permute(2, 0, 1)  # HWC to CHW

    im = img.type(torch.half) if fp_flag else img.type(torch.float)  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim
    return im

def do_detect(args, detection_network, img, img_size, stride, auto):
    # Do Detection Inference
    im_resize = preprocess_img(img, detection_network.fp16, img_size, stride, auto)
    pred = detection_network(im_resize, augment=False, visualize=False)

    # Detection NMS
    pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, None, False, max_det=args.max_det)

    # Process predictions
    for i, det in enumerate(pred):  # per predictions
        # Rescale boxes from img_size to img size
        det[:, :4] = scale_coords(im_resize.shape[2:], det[:, :4], img.shape).round()

    return det

def save_detection_result(args, det, names, p,mode, frame, imc):
    # prepare detection result path
    ### check detection Directories
    save_dir = increment_path(Path(args.project) / args.name, exist_ok=args.exist_ok)  # increment run
    (save_dir / 'labels' if args.save_bbox else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    p = Path(p)  # to Path
    save_path = str(save_dir / p.name)  # im.jpg
    txt_path = str(save_dir / 'labels' / p.stem) + ('' if mode == 'image' else f'_{frame}')  # im.txt
    gn = torch.tensor(imc.shape)[[1, 0, 1, 0]]  # normalization gain whwh
    annotator = Annotator(imc, line_width=3, example=str(names))


    for *xyxy, conf, cls in reversed(det):
        if args.save_bbox:  # Write to file
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if args.save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')

        if args.save-detect-img or args.save_crop:  # Add bbox to image
            c = int(cls)  # integer class
            label = None if args.hide_labels else (names[c] if args.hide_conf else f'{names[c]} {conf:.2f}')
            annotator.box_label(xyxy, label, color=colors(c, True))
            if args.save_crop:
                save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)

    # Stream results
    im0 = annotator.result()

    # Save results (image with detections)
    if args.save-detect-img:
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

    # Print detection results
    # s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if args.save_bbox else ''
