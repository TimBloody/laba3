import cv2
import numpy as np
def resize_img(img, max_width=512):
    h, w = img.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_size = (int(w * scale), int(h * scale))
        img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)
    return img


def create_red_mask(image_path, save_path=None):
    img = cv2.imread(image_path)
    img = resize_img(img, max_width=512)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # Красный цвет
    lower_red1 = np.array([0, 110, 50])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([140, 110, 50])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    kernel = np.ones((5, 5), np.uint8)
    red_mask = cv2.bitwise_or(mask1, mask2)
    closed = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel, iterations=4)
    
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(closed, connectivity=8)
    sizes = stats[1:, -1]  # пропускаем фон

    min_size = 10000  
    clean_mask = np.zeros_like(closed)
    # Дополнительная дилатация для закрытия внутренних дыр
    clean_mask = cv2.dilate(clean_mask, kernel, iterations=2)
    clean_mask = cv2.erode(clean_mask, kernel, iterations=2)

    for i in range(1, num_labels):
        if sizes[i - 1] > min_size:
            clean_mask[labels == i] = 255
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    cv2.imwrite(save_path, clean_mask)


#create_red_mask(r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\CNN\images\apples.jpg', r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\CNN\masks\mask2_red.png')
#create_red_mask(r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\CNN\images\apple1.jpeg', r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\CNN\masks\mask1_red.png')
