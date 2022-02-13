import cv2
import glob
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path

def make_percentile_color_samples(src_img: np.array, mask: np.array, percentile_list: list, sample_width: int = 100, gap: int = 10):
    src_img_height, src_img_width = src_img.shape[:2]
    dst_img_height = src_img_height
    dst_img_width = src_img_width + gap + sample_width
    dst_img = np.full((dst_img_height, dst_img_width, 3), 255, dtype='uint8')
    print(f'src_img : {src_img.shape}')
    print(f'dst_img : {dst_img.shape}')
    dst_img[:, :src_img_width] = src_img[:, :]

    # カラーサンプルの準備
    samples_num = len(percentile_list)
    sample_height = int((src_img_height - gap * (samples_num - 1)) / samples_num)
    sample_left_x = src_img_width + gap

    # グレースケール化してpercentilesで指定された明るさを得る
    src_img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    src_img_gray_masked = np.where(mask == 0, np.nan, src_img_gray)
    src_brightness_list = src_img_gray_masked.ravel()
    src_brightness_list_without_nan = src_brightness_list[np.isfinite(src_brightness_list)]
    percentile_brightness_list = np.percentile(src_brightness_list_without_nan, percentile_list)

    # 指定percentile毎に色見本を作成
    for i, bright in enumerate(percentile_brightness_list):
        # 特定の明るさの画素を抽出
        selected_pixcel = []
        for y in range(src_img_height):
            for x in range(src_img_width):
                if np.isfinite(src_img_gray_masked[y, x]) and (src_img_gray_masked[y, x] > bright - 2) and (src_img_gray_masked[y, x] < bright + 2):
                    selected_pixcel.append(src_img[y, x])

        # 外れ値を除外
        hsv = cv2.cvtColor(np.array([selected_pixcel]), cv2.COLOR_BGR2HSV)
        hue_list = hsv[:, :, 0][0]
        hue_thr = np.percentile(hue_list, [10, 90])
        print(f'hue threshold : {hue_thr}')
        legal_color_flg = list(np.array(hue_list >= hue_thr[0]) * np.array(hue_list <= hue_thr[1]))
        legal_color = np.array(selected_pixcel)[legal_color_flg]

        mean_color = np.mean(legal_color, axis=0)

        print(f'{i} : {bright} : {mean_color}')
        print()

        sample_top_y = (sample_height + gap) * (samples_num - i - 1)
        sample_bottom_y = sample_top_y + sample_height - 1
        cv2.rectangle(dst_img, (sample_left_x, sample_top_y), (dst_img_width - 1, sample_bottom_y), mean_color, thickness=-1)
        cv2.putText(
            dst_img,
            text=f'{percentile_list[i]}' + '% : ' + f'{int(bright)}',
            org=(sample_left_x + 5, sample_top_y + 10),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.3,
            color=(255, 0, 0)
        )

    return dst_img, percentile_brightness_list


if __name__ == '__main__':
    input_dir = './data/photo'
    mask_dir = './data/mask'
    output_dir = './output'

    img_file_list = glob.glob(f'{input_dir}/*.jpg')
    for im_file in img_file_list:
        im_name = Path(im_file).stem
        print('')
        print(f'path : {im_file}')
        img = cv2.imread(im_file)
        mask = cv2.imread(f'{mask_dir}/{im_name}.png', cv2.IMREAD_GRAYSCALE)
        output_img, percentile_brightness_list = make_percentile_color_samples(img, mask, [70, 75, 80, 85, 90, 95])
        # print(percentile_brightness_list)
        cv2.imwrite(f'{output_dir}/{im_name}.png', output_img)
