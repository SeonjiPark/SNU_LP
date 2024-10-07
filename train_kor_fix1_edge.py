import torch
from torch.utils.data import DataLoader

import logging
import os
from tqdm import tqdm
from nltk import edit_distance

from config import parse_args
from model import Model
from evaluate import eval
from utils.average_meter import AverageMeter
from utils.dataloader_kor import KorLP_Recognition_Dataset
from utils.datatransformer import AlignCollate
from utils.helpers import *
from utils.CTCConverter import *

# Set up GPU device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")  # 터미널에 현재 사용 중인 장치 표시

# Check how many GPUs are available
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
else:
    print("CUDA is not available. Running on CPU.")

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

######### Configuration #########
######### Configuration #########
######### Configuration #########
args = parse_args()

# Design Parameters
IMGH = args.imgH
IMGW = args.imgW
BATCH_MAX_LENGTH = args.batch_max_length
PAD = args.pad_image
IMG_COLOR = args.img_color
if IMG_COLOR == 'Gray':
    args.input_channel = 1
elif IMG_COLOR == 'RGB':
    args.input_channel = 3

# Session Parameters
GPU_NUM = args.gpu_num
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
N_EPOCHS = args.epochs

OPTIM_TYPE = args.optim_type
LR = args.lr
BETA1 = args.beta1
RHO = args.rho
EPS = args.eps
GRAD_CLIP = args.grad_clip

TRAIN_ACC_EVERY = args.train_acc_every
SAVE_EVERY = args.save_every
PRINT_EVERY = args.print_every
EVAL_EVERY = args.eval_every

# Directory Parameters
EXP_NAME = args.experiment_name
# EXP_NAME이 경로로 지정되어 있으면, 따로 'experiments/'를 추가하지 않음
if not os.path.isabs(EXP_NAME) and not EXP_NAME.endswith('/'):
    EXP_DIR = os.path.join('experiments', EXP_NAME)
else:
    EXP_DIR = EXP_NAME  # 이미 경로가 지정되어 있으면 추가하지 않음
DATA_DIR = args.data_dir
EXP_DIR = 'experiments/' + EXP_NAME
CKPT_DIR = os.path.join(EXP_DIR, args.ckpt_dir)
LOG_DIR = os.path.join(EXP_DIR, args.log_dir)
WEIGHTS = args.weights
BEST_WEIGHTS = args.best_weights

# Check if directory does not exist
create_path(EXP_DIR)
create_path(CKPT_DIR)
create_path(LOG_DIR)
create_path(os.path.join(LOG_DIR, 'train'))
create_path(os.path.join(LOG_DIR, 'test'))

# Set up logger
filename = os.path.join(LOG_DIR, 'logs.txt')
logging.basicConfig(filename=filename,format='[%(levelname)s] %(asctime)s %(message)s')
logging.getLogger().setLevel(logging.INFO)

for key,value in sorted((args.__dict__).items()):
    print('\t%15s:\t%s' % (key, value))
    logging.info('\t%15s:\t%s' % (key, value))

######### Configuration #########
######### Configuration #########
######### Configuration #########

# Set up Dataset
converter = CTCLabelConverter(chars)
args.num_class = len(converter.character)

train_dataset = KorLP_Recognition_Dataset(DATA_DIR, 'Training', IMG_COLOR, args.add_noise, args.noise_var, args.noise_amount)
test_dataset = KorLP_Recognition_Dataset(DATA_DIR, 'Validation', IMG_COLOR)

Collate = AlignCollate(IMGH, IMGW, PAD)

train_dataloader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=True,
    drop_last=True,
    collate_fn=Collate
)

test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    num_workers=NUM_WORKERS,
    shuffle=False,
    drop_last=False,
    collate_fn=Collate
)

# Set up GPU
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_NUM)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Network declare
network = Model(args, device)

network = network.to(device)
network = initialize_model(network)
filtered_parameters = filter_parameter(network)

# Set up Loss Functions
criterion = torch.nn.CTCLoss(zero_infinity=True).to(device)

# Load the pretrained model if exists
init_epoch = 0
best_metric = 0
best_distance = 1000

if os.path.exists(os.path.join(CKPT_DIR, WEIGHTS)):
    logging.info('Recovering from %s ...' % os.path.join(CKPT_DIR, WEIGHTS))
    checkpoint = torch.load(os.path.join(CKPT_DIR, WEIGHTS))
    init_epoch = checkpoint['epoch_idx']
    best_metric = checkpoint['best_metric']
    best_distance = checkpoint['best_distance']
    LR = checkpoint['lr']
    network.load_state_dict(checkpoint['network'])
    logging.info('Recover completed. Current epoch = #%d' % (init_epoch))

# Create Optimizer
if OPTIM_TYPE == 'Adam':
    optimizer = torch.optim.Adam(filtered_parameters, lr=LR, betas=(BETA1, 0.999))
elif OPTIM_TYPE == 'Adadelta':
    optimizer = torch.optim.Adadelta(filtered_parameters, lr=LR, rho=RHO, eps=EPS)

early_stop_patience = 20  # 성능 개선이 없을 때 멈추기 위한 최대 에포크 수
no_improvement_epochs = 0  # 성능 개선이 없는 에포크 수

# 초기 best_metric 값 설정
best_metric = 0

for epoch_idx in range(init_epoch+1, N_EPOCHS):

    # Metric holders
    losses = AverageMeter()

    # Network to train mode
    network.train()

    total_train_samples = 0
    train_correct_samples = 0
    train_avg_distance = 0

    # Train for batches
    for batch_idx, data in enumerate(tqdm(train_dataloader)):

        imgs, labels = data
        imgs = imgs.to(torch.float32)
        
        # Edge enhancement 적용
        if args.add_edge:
            imgs = enhance_edges(imgs, args.edge_amount)

        texts, lengths = converter.encode(labels, batch_max_length=BATCH_MAX_LENGTH, device=device)

        # Data to cuda
        imgs = imgs.to(device)
        
        preds = network(imgs, texts)

        preds_size = torch.IntTensor([preds.size(1)] * BATCH_SIZE)
        preds_softmax = preds.log_softmax(2).permute(1, 0, 2)
        loss = criterion(preds_softmax, texts, preds_size, lengths)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(network.parameters(), 5)
        optimizer.step()

        losses.update(loss.item())

        if epoch_idx % TRAIN_ACC_EVERY == 0:
            SAMPLE_NUM = imgs.size(0)
            _, preds_index = preds.max(2)
            decoded = converter.decode(preds_index, preds_size)

            for idx in range(SAMPLE_NUM):
                pred_text = decoded[idx]

                # 먼저 pred_text의 길이가 충분히 긴지 확인 (최소 5자)
                if len(pred_text) >= 5:
                    # 뒤에서 다섯 번째 자리가 숫자인 경우 문자로 대체
                    if pred_text[-5].isdigit():
                        char_indices = [i for i, c in enumerate(chars) if c.isalpha()]
                        probs = preds_softmax[:, idx, char_indices]
                        max_prob_idx = probs.argmax(dim=1)[-5]
                        pred_char = chars[char_indices[max_prob_idx]]
                        pred_text = pred_text[:-5] + pred_char + pred_text[-4:]

                # 마지막 4자리를 처리하기 전에 길이 확인
                if len(pred_text) >= 4:
                    for i in range(-4, 0):
                        if pred_text[i].isalpha():
                            num_indices = [i for i, c in enumerate(chars) if c.isdigit()]
                            probs = preds_softmax[:, idx, num_indices]
                            max_prob_idx = probs.argmax(dim=1)[i]
                            pred_char = chars[num_indices[max_prob_idx]]
                            pred_text = pred_text[:i] + pred_char + pred_text[i+1:]

                if labels[idx] == pred_text:
                    train_correct_samples += 1
                total_train_samples += 1
                train_avg_distance += edit_distance(labels[idx], pred_text)

    # 정확도 계산 시 total_train_samples가 0이 아닌지 확인
    if total_train_samples > 0:
        train_acc = train_correct_samples / total_train_samples
        train_avg_distance /= total_train_samples
    else:
        train_acc = 0
        train_avg_distance = 0
        print("Warning: No training samples processed in this epoch.")

    # Print loss, accuracy for this epoch
    print(f'[Epoch {epoch_idx}/{N_EPOCHS}] Loss: {losses.avg():.4f}, Accuracy: {train_acc*100:.2f}%')

    if epoch_idx % PRINT_EVERY == 0:
        logging.info(f'[Epoch {epoch_idx}/{N_EPOCHS}] Loss: {losses.avg():.4f}, Accuracy: {train_acc*100:.2f}%')

    if epoch_idx % EVAL_EVERY == 0:
        test_acc, correct_sample, total_samples, avg_distance = eval(network, test_dataloader, device, converter, BATCH_MAX_LENGTH)

        # Output evaluation results
        print(f'====== Evaluation Accuracy: {test_acc*100:.2f}%, Edit Distance: {avg_distance:.2f}')
        logging.info(f'====== Evaluation Accuracy: {test_acc*100:.2f}%, Edit Distance: {avg_distance:.2f}')

        if test_acc > best_metric:
            best_metric = test_acc
            best_distance = avg_distance
            no_improvement_epochs = 0  # 성능이 개선되면 초기화

            output_path = os.path.join(CKPT_DIR, BEST_WEIGHTS)
            torch.save({
                'best_metric': best_metric,
                'best_distance': best_distance,
                'network': network.state_dict()
            }, output_path)

            print('Best Model Saved')
            logging.info('Best Model Saved')
        else:
            no_improvement_epochs += 1  # 성능 개선이 없으면 증가

        logging.info(f'====== Best Accuracy = {best_metric*100:.2f}%, Best Distance = {best_distance:.2f}')

    # Check for early stopping
    if no_improvement_epochs >= early_stop_patience:
        print(f'Early stopping triggered after {early_stop_patience} epochs with no improvement.')
        logging.info(f'Early stopping triggered after {early_stop_patience} epochs with no improvement.')
        break

    if epoch_idx % SAVE_EVERY == 0:
        output_path = os.path.join(CKPT_DIR, WEIGHTS)
        torch.save({
            'epoch_idx': epoch_idx,
            'lr': optimizer.param_groups[0]["lr"],
            'best_metric': best_metric,
            'best_distance': best_distance,
            'network': network.state_dict()
        }, output_path)
        print('Model Saved')
        logging.info('Model Saved')
