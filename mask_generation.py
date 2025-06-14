import cv2
import numpy as np
import os
from skimage.segmentation import watershed
from skimage.morphology import local_maxima
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.filters import rank
import matplotlib.pyplot as plt

class MaskGenerator:
    def __init__(self, images_dir=r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\images', masks_dir=r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\masks'):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        
        # Создаем директорию для масок если её нет
        os.makedirs(masks_dir, exist_ok=True)
    
    def generate_mask_watershed(self, image_path):
        """
        Генерация маски с помощью алгоритма водораздела
        """
        # Загружаем изображение
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Конвертируем в RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Конвертируем в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Нормализация яркости
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        gray = clahe.apply(gray)
        
        # Применяем медианный фильтр для уменьшения шума
        denoised = cv2.medianBlur(gray, 5)
        
        # Применяем пороговую обработку Оцу
        _, binary = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Морфологические операции для очистки
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Находим фон
        sure_bg = cv2.dilate(closing, kernel, iterations=3)
        
        # Находим передний план
        dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
        dist_transform = cv2.normalize(dist_transform, None, 0, 1.0, cv2.NORM_MINMAX)
        
        # Адаптивный порог для sure_fg
        _, sure_fg = cv2.threshold(dist_transform, 0.6, 1.0, 0)
        sure_fg = np.uint8(sure_fg * 255)
        
        # Находим неопределенную область
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Создаем маркеры для водораздела
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Применяем алгоритм водораздела
        markers = cv2.watershed(image_rgb, markers)
        
        # Создаем финальную маску
        mask = np.zeros_like(gray)
        mask[markers > 1] = 255
        mask[markers == 1] = 0
        
        # Постобработка маски
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return mask
    
    def generate_mask_adaptive(self, image_path):
        """
        Альтернативный метод генерации маски с адаптивной пороговой обработкой
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Не удалось загрузить изображение: {image_path}")
        
        # Конвертируем в градации серого
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Применяем Гауссовский фильтр
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Адаптивная пороговая обработка
        adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, 
                                              cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                              cv2.THRESH_BINARY_INV, 11, 2)
        
        # Морфологические операции
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        
        return closing
    
    def process_all_images(self, method='watershed'):
        """
        Обрабатывает все изображения в директории
        """
        if not os.path.exists(self.images_dir):
            print(f"Директория {self.images_dir} не существует!")
            return
        
        image_files = [f for f in os.listdir(self.images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"В директории {self.images_dir} нет изображений!")
            return
        
        print(f"Найдено {len(image_files)} изображений для обработки...")
        
        for image_file in image_files:
            try:
                image_path = os.path.join(self.images_dir, image_file)
                print(f"Обрабатываем: {image_file}")
                
                # Генерируем маску
                if method == 'watershed':
                    mask = self.generate_mask_watershed(image_path)
                else:
                    mask = self.generate_mask_adaptive(image_path)
                
                # Сохраняем маску
                mask_filename = os.path.splitext(image_file)[0] + '_mask.png'
                mask_path = os.path.join(self.masks_dir, mask_filename)
                cv2.imwrite(mask_path, mask)
                
                print(f"Маска сохранена: {mask_filename}")
                
            except Exception as e:
                print(f"Ошибка при обработке {image_file}: {str(e)}")
        
        print("Генерация масок завершена!")
    
    def visualize_result(self, image_file):
        """
        Визуализация результата для одного изображения
        """
        image_path = os.path.join(self.images_dir, image_file)
        mask_file = os.path.splitext(image_file)[0] + '_mask.png'
        mask_path = os.path.join(self.masks_dir, mask_file)
        
        if not os.path.exists(image_path) or not os.path.exists(mask_path):
            print("Файлы не найдены!")
            return
        
        # Загружаем изображение и маску
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Создаем визуализацию
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image_rgb)
        axes[0].set_title('Исходное изображение')
        axes[0].axis('off')
        
        axes[1].imshow(mask, cmap='gray')
        axes[1].set_title('Маска')
        axes[1].axis('off')
        
        # Накладываем маску на изображение
        overlay = image_rgb.copy()
        overlay[mask == 255] = [255, 0, 0]  # Красный цвет для объекта
        result = cv2.addWeighted(image_rgb, 0.7, overlay, 0.3, 0)
        
        axes[2].imshow(result)
        axes[2].set_title('Результат сегментации')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Создаем генератор масок
    mask_gen = MaskGenerator()
    
    # Обрабатываем все изображения
    mask_gen.process_all_images(method='watershed')
    
    # Если нужно посмотреть результат для конкретного изображения
    # mask_gen.visualize_result('apple1.jpeg')