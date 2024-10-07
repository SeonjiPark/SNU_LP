import os
import shutil
import random

# 경로 설정
validation_image_dir = "./DATASET/kamo_lp_80/Validation/image"  # Validation 폴더의 이미지 경로
validation_label_dir = "./DATASET/kamo_lp_80/Validation/label"  # Validation 폴더의 라벨 경로 (.json 파일)

training_image_dir = "./DATASET/kamo_lp_80/Training/image"  # Training 폴더의 이미지 경로
training_label_dir = "./DATASET/kamo_lp_80/Training/label"  # Training 폴더의 라벨 경로 (.json 파일)

# Validation 폴더에서 이미지 파일 리스트 얻기
image_files = os.listdir(validation_image_dir)
image_files = [f for f in image_files if os.path.isfile(os.path.join(validation_image_dir, f))]

# 무작위로 200개의 이미지 선택
random_selected_images = random.sample(image_files, 200)

# 이미지 파일과 대응되는 .json 라벨 파일을 Training 폴더로 이동
for image_file in random_selected_images:
    # 이미지 파일 경로 설정
    src_image_path = os.path.join(validation_image_dir, image_file)
    dst_image_path = os.path.join(training_image_dir, image_file)
    
    # 대응되는 라벨 파일 이름 (확장자만 .json으로 바꿔줍니다.)
    label_file = os.path.splitext(image_file)[0] + ".json"
    src_label_path = os.path.join(validation_label_dir, label_file)
    dst_label_path = os.path.join(training_label_dir, label_file)
    
    # 이미지 및 라벨 파일을 Training 폴더로 이동
    shutil.move(src_image_path, dst_image_path)
    shutil.move(src_label_path, dst_label_path)

print(f"{len(random_selected_images)} images and their corresponding JSON labels have been moved to the Training folder.")
