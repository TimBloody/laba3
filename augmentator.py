import cv2
import numpy as np
import os
import random
from scipy.ndimage import rotate
from skimage import transform
from skimage.util import random_noise
import albumentations as A
from albumentations.pytorch import ToTensorV2
class DataAugmentator:
    def __init__(self, images_dir=r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\images', 
                 masks_dir=r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\masks', 
                 aug_images_dir=r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\augmented_images', 
                 aug_masks_dir=r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\augmented_masks'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.aug_images_dir = aug_images_dir
        self.aug_masks_dir = aug_masks_dir
        
        # Создаем директории для аугментированных данных
        os.makedirs(aug_images_dir, exist_ok=True)
        os.makedirs(aug_masks_dir, exist_ok=True)
        
        # Определяем трансформации
        self.transform = A.Compose([
            A.RandomRotate90(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, p=0.3),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.RandomResizedCrop(size=(256, 256), scale=(0.8, 1.0), ratio=(0.9, 1.1), p=0.5),
        ])
    
    def manual_augmentations(self, image, mask):
        """
        Ручные аугментации без использования albumentations
        """
        augmented_pairs = []
        
        # 1. Поворот на случайный угол (с интерполяцией для масок)
        for angle in [15, 30, 45, -15, -30]:
            rotated_img = self.rotate_image(image, angle)
            rotated_mask = self.rotate_image(mask, angle, is_mask=True)
            augmented_pairs.append((rotated_img, rotated_mask, f'rot_{angle}'))
        
        # 2. Отражения
        # Горизонтальное отражение
        flipped_h_img = cv2.flip(image, 1)
        flipped_h_mask = cv2.flip(mask, 1)
        augmented_pairs.append((flipped_h_img, flipped_h_mask, 'flip_h'))
        
        # Вертикальное отражение
        flipped_v_img = cv2.flip(image, 0)
        flipped_v_mask = cv2.flip(mask, 0)
        augmented_pairs.append((flipped_v_img, flipped_v_mask, 'flip_v'))
        
        # 3. Изменение яркости и контраста (только для изображения)
        bright_img = self.adjust_brightness_contrast(image, brightness=20, contrast=1.1)
        augmented_pairs.append((bright_img, mask, 'bright'))
        
        dark_img = self.adjust_brightness_contrast(image, brightness=-20, contrast=0.9)
        augmented_pairs.append((dark_img, mask, 'dark'))
        
        # 4. Размытие (только для изображения)
        blurred_img = cv2.GaussianBlur(image, (3, 3), 0)
        augmented_pairs.append((blurred_img, mask, 'blur'))
        
        # 5. Добавление шума (только для изображения)
        noisy_img = self.add_noise(image, noise_level=15)
        augmented_pairs.append((noisy_img, mask, 'noise'))
        
        return augmented_pairs
    
    def rotate_image(self, image, angle, is_mask=False):
        """Поворот изображения на заданный угол"""
        if len(image.shape) == 3:
            rotated = rotate(image, angle, axes=(1, 0), reshape=False, 
                            order=1 if not is_mask else 0, 
                            mode='constant', cval=0)
        else:
            rotated = rotate(image, angle, reshape=False, 
                            order=0, mode='constant', cval=0)
        return rotated.astype(image.dtype)
    
    def adjust_brightness_contrast(self, image, brightness=0, contrast=1.0):
        """Изменение яркости и контраста"""
        adjusted = cv2.convertScaleAbs(image, alpha=contrast, beta=brightness)
        return adjusted
    
    def add_noise(self, image, noise_level=15):
        """Добавление шума к изображению"""
        noise = np.random.normal(0, noise_level, image.shape).astype(np.uint8)
        noisy_image = cv2.add(image, noise)
        return np.clip(noisy_image, 0, 255).astype(np.uint8)
    
    def scale_image(self, image, mask, scale_factor):
        """Масштабирование изображения и маски"""
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        # Масштабируем изображение
        scaled_img = cv2.resize(image, (new_width, new_height))
        scaled_mask = cv2.resize(mask, (new_width, new_height))
        
        # Если увеличили, обрезаем до исходного размера
        if scale_factor > 1:
            start_x = (new_width - width) // 2
            start_y = (new_height - height) // 2
            scaled_img = scaled_img[start_y:start_y+height, start_x:start_x+width]
            scaled_mask = scaled_mask[start_y:start_y+height, start_x:start_x+width]
        
        # Если уменьшили, добавляем отступы
        elif scale_factor < 1:
            pad_x = (width - new_width) // 2
            pad_y = (height - new_height) // 2
            scaled_img = cv2.copyMakeBorder(scaled_img, pad_y, height-new_height-pad_y, 
                                          pad_x, width-new_width-pad_x, cv2.BORDER_CONSTANT, value=0)
            scaled_mask = cv2.copyMakeBorder(scaled_mask, pad_y, height-new_height-pad_y, 
                                           pad_x, width-new_width-pad_x, cv2.BORDER_CONSTANT, value=0)
        
        return scaled_img, scaled_mask
    
    def augment_with_albumentations(self, image, mask):
        """Аугментация с использованием albumentations"""
        augmented_pairs = []
        
        # Применяем трансформации несколько раз для большего разнообразия
        for i in range(5):
            try:
                transformed = self.transform(image=image, mask=mask)
                aug_image = transformed['image']
                aug_mask = transformed['mask']
                augmented_pairs.append((aug_image, aug_mask, f'albu_{i}'))
            except Exception as e:
                print(f"Ошибка при аугментации albumentations: {e}")
                continue
        
        return augmented_pairs
    
    def process_dataset(self, multiplier=5, use_albumentations=True):
        """
        Обрабатывает весь датасет и создает аугментированные версии
        multiplier - во сколько раз увеличить датасет
        """
        # Получаем список всех изображений
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"В директории {self.images_dir} нет изображений!")
            return
        
        print(f"Найдено {len(image_files)} изображений для аугментации...")
        
        # Получаем список всех масок
        mask_files = [f for f in os.listdir(self.masks_dir) 
                     if f.lower().endswith('.png')]
        
        # Сортируем файлы, чтобы обеспечить правильное соответствие
        image_files.sort()
        mask_files.sort()
        
        if len(image_files) != len(mask_files):
            print(f"Внимание: количество изображений ({len(image_files)}) не совпадает с количеством масок ({len(mask_files)})")
        
        # Создаем пары изображение-маска
        # Берем первую маску для первого изображения, вторую для второго и т.д.
        image_mask_pairs = list(zip(image_files, mask_files))
        
        total_generated = 0
        
        for image_file, mask_file in image_mask_pairs:
            try:
                # Путь к изображению и маске
                image_path = os.path.join(self.images_dir, image_file)
                mask_path = os.path.join(self.masks_dir, mask_file)
                
                print(f"Обрабатываем пару: {image_file} с маской {mask_file}")
                
                # Загружаем изображение и маску
                image = cv2.imread(image_path)
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                
                if image is None or mask is None:
                    print(f"Не удалось загрузить {image_file} или его маску")
                    continue
                
                
                
                # Сначала сохраняем оригинал в папку аугментированных данных
                orig_img_name = f"orig_{image_file}"
                orig_mask_name = f"orig_{mask_file}"
                
                cv2.imwrite(os.path.join(self.aug_images_dir, orig_img_name), image)
                cv2.imwrite(os.path.join(self.aug_masks_dir, orig_mask_name), mask)
                
                # Генерируем аугментации
                all_augmentations = []
                
                # Ручные аугментации
                manual_augs = self.manual_augmentations(image, mask)
                all_augmentations.extend(manual_augs)
                
                # Аугментации с albumentations (если доступно)
                if use_albumentations:
                    try:
                        albu_augs = self.augment_with_albumentations(image, mask)
                        all_augmentations.extend(albu_augs)
                    except Exception as e:
                        print(f"Ошибка albumentations для {image_file}: {e}")
                
                # Выбираем случайные аугментации до нужного количества
                random.shuffle(all_augmentations)
                selected_augs = all_augmentations[:multiplier-1]  # -1 потому что оригинал уже сохранен
                
                # Сохраняем аугментированные изображения
                for idx, (aug_img, aug_mask, aug_type) in enumerate(selected_augs):
                    base_name = os.path.splitext(image_file)[0]
                    mask_base_name = os.path.splitext(mask_file)[0]
                    
                    aug_img_name = f"{base_name}_{aug_type}_{idx}.png"
                    aug_mask_name = f"{mask_base_name}_{aug_type}_{idx}.png"
                    
                    # Убеждаемся, что изображения в правильном формате
                    if aug_img.dtype != np.uint8:
                        aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
                    if aug_mask.dtype != np.uint8:
                        aug_mask = np.clip(aug_mask, 0, 255).astype(np.uint8)
                    
                    cv2.imwrite(os.path.join(self.aug_images_dir, aug_img_name), aug_img)
                    cv2.imwrite(os.path.join(self.aug_masks_dir, aug_mask_name), aug_mask)
                    
                    total_generated += 1
                
                print(f"Создано {len(selected_augs) + 1} вариантов для {image_file}")
                
            except Exception as e:
                print(f"Ошибка при обработке {image_file}: {str(e)}")
        
        print(f"\nАугментация завершена!")
        print(f"Всего создано {total_generated + len(image_files)} изображений")
        print(f"Исходных: {len(image_files)}")
        print(f"Аугментированных: {total_generated}")
    
    def get_dataset_info(self):
        """Получение информации о датасете"""
        orig_images = len([f for f in os.listdir(self.images_dir) 
                          if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        orig_masks = len([f for f in os.listdir(self.masks_dir) 
                         if f.lower().endswith('.png')])
        
        aug_images = len([f for f in os.listdir(self.aug_images_dir) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))])
        aug_masks = len([f for f in os.listdir(self.aug_masks_dir) 
                        if f.lower().endswith('.png')])
        
        print(f"\n=== Информация о датасете ===")
        print(f"Исходные изображения: {orig_images}")
        print(f"Исходные маски: {orig_masks}")
        print(f"Аугментированные изображения: {aug_images}")
        print(f"Аугментированные маски: {aug_masks}")
        print(f"Общий размер датасета: {aug_images} пар изображение-маска")

    def validate_mask(self, mask):
        """Проверка качества маски"""
        # Проверка на пустую маску
        if np.sum(mask) == 0:
            return False
        
        # Проверка на слишком маленькие области
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        
        # Проверка на слишком большие области (больше 90% изображения)
        if np.sum(mask) > 0.9 * mask.size * 255:
            return False
        
        return True

if __name__ == "__main__":
    # Создаем аугментатор
    augmentator = DataAugmentator()
    
    # Показываем информацию о текущем датасете
    augmentator.get_dataset_info()
    
    # Выполняем аугментацию (увеличиваем датасет в 8 раз)
    augmentator.process_dataset(multiplier=8, use_albumentations=False)
    
    # Показываем финальную информацию
    augmentator.get_dataset_info()