import os
import json

def encode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province[idx]
        if prov in text:
            text = text.replace(prov, province_replace[idx])

    return text

def decode_province(text, province, province_replace):

    for idx in range(len(province)):
        prov = province_replace[idx]
        if prov in text:
            text = text.replace(prov, province[idx])
    return text

def isKorean(s):
    return not s.isascii()

def get_chars(DATA_DIR, subset, province, province_replace):

    IMG_DIR = 'image'
    LABEL_DIR = 'label'

    # Read Label path
    label_path = []

    for file in os.listdir(os.path.join(DATA_DIR, subset, LABEL_DIR)):
        if file.endswith('.json'):
            label_path.append(file)

    label_path = sorted(label_path)

    img_paths = []
    labels = []
    total_num = 0

    # Obtain Image & Label without provinces
    for l_path in label_path:
        cur_path = os.path.join(DATA_DIR, subset, LABEL_DIR, l_path)
        f = json.load(open(cur_path))


        if not '-' in f['value']:
            if not '미주홀' in f['value']:
                if ' ' in f['value']:
                    fixed = f['value'].replace(' ','')
                    fixed = encode_province(fixed, province, province_replace)
                    img_paths.append(f['imagePath'])
                    labels.append(fixed)
                else:
                    fixed = encode_province(f['value'], province, province_replace)
                    img_paths.append(f['imagePath'])
                    labels.append(fixed)

        
        total_num += 1

    # Obtain character list 'without' provinces
    chars = []
    max_length = 0

    for idx, label in enumerate(labels):
        cur_chars = list(label)
        
        for char in cur_chars:
            if not char in chars:
                chars.append(char)

        if len(cur_chars) > max_length:
            max_length = len(cur_chars)

    chars = sorted(chars)

    return chars, max_length

def get_provinces(DATA_DIR, subset):

    provinces = []

    IMG_DIR = 'image'
    LABEL_DIR = 'label'

    # Read Label path
    label_path = []

    for file in os.listdir(os.path.join(DATA_DIR, subset, LABEL_DIR)):
        if file.endswith('.json'):
            label_path.append(file)

    label_path = sorted(label_path)

    provinces = []

    # Obtain Image & Label without provinces
    for l_path in label_path:
        cur_path = os.path.join(DATA_DIR, subset, LABEL_DIR, l_path)
        f = json.load(open(cur_path))

        label = f['value']

        cur_province = ""
        is_province = False

        for cur_char in label:
            if is_province and isKorean(cur_char):
                cur_province += cur_char
            elif not is_province and isKorean(cur_char):
                cur_province += cur_char
                is_province = True
            elif is_province and not isKorean(cur_char):
                break

        if len(cur_province) > 1:
            if not cur_province in provinces:
                provinces.append(cur_province)

    return provinces

if __name__ == '__main__':

    STEP1_DONE = True
    if not STEP1_DONE:
        ###################################
        #    1. Get Raw Provinces         #
        ###################################
        train_province = get_provinces('../DATASET/KorLP', 'Training')
        val_province = get_provinces('../DATASET/KorLP', 'Validation')

        provinces_raw = sorted(list(set(train_province + val_province)))
        # provinces_raw = ['강원', '경기', '경기고양나', '경기고양다', '경기고양사', '경기고양타', '경기광명파', 
        # '경기김포자', '경기부천다', '경기부천바', '경기부천사', '경기부천아', '경기부천자', '경기부천차', 
        # '경기부천카', '경기부천타', '경기부천파', '경기부천하', '경기안산라', '경기안양바', '경기인천아', 
        # '경남', '경북', '광주', '대구', '대구서하', '대전', '서울', '서울강남타', '서울강남파', '서울강남하',
        # '서울강서자', '서울강서차', '서울관악카', '서울관악파', '서울구로가', '서울구로자', '서울구로카', 
        # '서울금천자', '서울동대문파', '서울동작자', '서울서대문다', '서울서대문마', '서울서대문바', '서울서초마', 
        # '서울양천마', '서울양천자', '서울양천차', '서울영등포아', '서울용산자', '영경기', '영서울', '영인천', '인천', 
        # '인천계양사', '인천남동파', '인천미주홀파', '인천미추홀파', '인천부평사', '인천부평아', '인천서타', '인천연수나', '인천연수다', 
    # '인천중마', '전남', '전북', '충남', '충북']

    ###################################
    #  2. Process Province Manually   #
    ###################################
    province = ['대구서', '동대문', '미추홀', '서대문', '영등포', '인천서', '인천중',
                        '강남', '강서', '강원', '경기', '경남', '경북', '계양', '고양', '관악', '광명', '광주', '구로', '금천', '김포', '남동', 
                        '대구', '대전', '동작', '부천', '부평', '서울', '서초', '안산', '안양', '양천', '연수', '용산', '인천', '전남', '전북', 
                        '충남', '충북', '영']

    province_replace = ['괅', '놝', '돩', '랅', '맑', '밝', '삵', '앍', '잙', '찱',
                        '괉', '놡', '돭', '랉', '맕', '밡', '삹', '앑', '잝', '찵',
                        '괋', '놣', '돯', '뢇', '맗', '밣', '삻', '앓', '잟', '찷',
                        '괇', '놟', '돫', '뢃', '맓', '밟', '삷', '앏', '잛', '찳']


    ##########################################
    #  3. Get Total Character Num / Length   #
    ##########################################
    t_chars, t_max_length = get_chars('../DATASET/KorLP', 'Training', province, province_replace)
    v_chars, v_max_length = get_chars('../DATASET/KorLP', 'Validation', province, province_replace)

    print(sorted(t_chars))
    print('Training Char num : ', len(t_chars))
    print('Max Length : ', t_max_length)

    print(sorted(v_chars))
    print('Validation Char num : ', len(v_chars))
    print('Max Length : ', v_max_length)

    total_chars = sorted(list(set(t_chars) | set(v_chars)))
    total_max_length = max(t_max_length, v_max_length)

    print(total_chars)
    print('Total Char num : ', len(total_chars))
    print('Total Max Length : ', total_max_length)

    char_w_province = sorted([x for x in total_chars if x not in province_replace])
    print(char_w_province)
    print(len(char_w_province))