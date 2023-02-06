import cv2
from pathlib import Path
from tqdm import tqdm

from fvr import Finger_crop

fvr_path = Path("/home/kevinhuang/github/articulated-animation/log/fvusm256-gif 31_10_22_03.42.06/reconstruction/png")
save_path = "/home/kevinhuang/github/articulated-animation/log/fvusm256-gif 31_10_22_03.42.06/reconstruction_1218_crop"

fvr_list = list(fvr_path.rglob("*.png"))

for fvr in tqdm(fvr_list):
    img = cv2.imread(fvr.as_posix(), cv2.IMREAD_GRAYSCALE)
    img = img[128 - 96: 128 + 96, 128 - 64: 128 + 64] # H, W
    img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # cv2.imwrite(f'cut.png', img)
    # break

    roi_extract = Finger_crop(output_w=144, output_h=64, dataset='FVUSM')
    roi_img, edges = roi_extract.crop_finger(img)
    cv2.imwrite(f'/home/kevinhuang/github/articulated-animation/roi/{fvr.name}', roi_img)
    