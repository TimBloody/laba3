import os
import sys
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Импортируем наши модули
from mask_generation import MaskGenerator
from augmentator import DataAugmentator
from CNN import SegmentationTrainer, create_data_loaders

def setup_directories():
    """Создание необходимых директорий"""
    directories = [
        r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\images',
        r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\masks',
        r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\augmented_images',
        r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\augmented_masks',
        r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\test',
        r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\result'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Директория {directory} создана/проверена")

def check_dataset():
    """Проверка наличия данных в датасете"""
    images_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\images'
    masks_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\masks'
    
    if not os.path.exists(images_dir):
        print(f"❌ Директория {images_dir} не существует!")
        return False
    
    image_files = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not image_files:
        print(f"❌ В директории {images_dir} нет изображений!")
        print("Поместите изображения (например, apple1.jpeg, apple2.jpeg) в папку dataset/images/")
        return False
    
    print(f"✅ Найдено {len(image_files)} изображений в {images_dir}")
    
    # Проверяем маски
    if os.path.exists(masks_dir):
        mask_files = [f for f in os.listdir(masks_dir) if f.endswith('_red.png')]
        print(f"✅ Найдено {len(mask_files)} масок в {masks_dir}")
    
    return True

def step1_generate_masks():
    """Шаг 1: Генерация масок"""
    print("\n" + "="*50)
    print("ШАГ 1: ГЕНЕРАЦИЯ МАСОК")
    print("="*50)
    
    try:
        # Создаем генератор масок
        mask_gen = MaskGenerator()
        
        # Генерируем маски для всех изображений
        mask_gen.process_all_images(method='watershed')
        
        print("✅ Генерация масок завершена!")
        return True
    except Exception as e:
        print(f"❌ Ошибка при генерации масок: {str(e)}")
        return False

def step2_augment_data(multiplier=10):
    """Шаг 2: Аугментация данных"""
    print("\n" + "="*50)
    print("ШАГ 2: АУГМЕНТАЦИЯ ДАННЫХ")
    print("="*50)
    
    try:
        # Создаем аугментатор
        augmentator = DataAugmentator()
        
        # Показываем информацию о текущем датасете
        augmentator.get_dataset_info()
        
        # Выполняем аугментацию
        augmentator.process_dataset(multiplier=multiplier, use_albumentations=False)
        
        # Показываем финальную информацию
        augmentator.get_dataset_info()
        
        print("✅ Аугментация данных завершена!")
        return True
    except Exception as e:
        print(f"❌ Ошибка при аугментации данных: {str(e)}")
        return False

def step3_train_model(model_type='unet', epochs=30, batch_size=4):
    """Шаг 3: Обучение нейронной сети"""
    print("\n" + "="*50)
    print("ШАГ 3: ОБУЧЕНИЕ НЕЙРОННОЙ СЕТИ")
    print("="*50)
    
    # Пути к аугментированным данным
    images_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\augmented_images'
    masks_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\augmented_masks'

    # Проверяем наличие аугментированных данных
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        print("❌ Аугментированные данные не найдены! Запустите сначала аугментацию.")
        return False, None
    
    try:
        # Создаем data loaders
        print("Создание data loaders...")
        train_loader, val_loader = create_data_loaders(
            images_dir, masks_dir, 
            batch_size=batch_size, 
            val_split=0.2,
            img_size=(256, 256)
        )
        
        # Создаем тренер
        trainer = SegmentationTrainer(model_type=model_type, device='auto')
        
        # Обучаем модель
        print(f"Начинаем обучение модели {model_type}...")
        model = trainer.train(train_loader, val_loader, epochs=epochs, lr=0.001)
        
        # Сохраняем модель
        model_path = f'trained_{model_type}_model.pth'
        trainer.save_model(model_path)
        
        # Показываем графики обучения
        trainer.plot_training_history()
        
        print("✅ Обучение модели завершено!")
        return True, trainer
        
    except Exception as e:
        print(f"❌ Ошибка при обучении модели: {str(e)}")
        return False, None

def step4_test_model(trainer, test_dir='test'):
    """Шаг 4: Тестирование модели"""
    print("\n" + "="*50)
    print("ШАГ 4: ТЕСТИРОВАНИЕ МОДЕЛИ")
    print("="*50)
    
    if not os.path.exists(test_dir):
        print(f"❌ Директория {test_dir} не существует!")
        return False
    
    # Получаем список тестовых изображений
    test_images = [f for f in os.listdir(test_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    if not test_images:
        print(f"❌ В директории {test_dir} нет изображений для тестирования!")
        print("Поместите тестовые изображения в папку test/")
        return False
    
    print(f"Найдено {len(test_images)} тестовых изображений")
    
    # Создаем директорию для результатов
    results_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\result'
    os.makedirs(results_dir, exist_ok=True)
    
    # Тестируем на каждом изображении
    for i, test_image in enumerate(test_images):
        test_path = os.path.join(test_dir, test_image)
        
        print(f"Обрабатываем: {test_image}")
        
        try:
            # Получаем предсказание
            mask_binary, mask_prob = trainer.predict(test_path)
            
            # Сохраняем результат
            result_name = os.path.splitext(test_image)[0] + '_predicted_mask.png'
            result_path = os.path.join(results_dir, result_name)
            cv2.imwrite(result_path, mask_binary)
            
            # Создаем визуализацию
            visualize_prediction(test_path, mask_binary, mask_prob, 
                               os.path.join(results_dir, f'visualization_{i+1}.png'))
            
            print(f"✅ Результат сохранен: {result_name}")
            
        except Exception as e:
            print(f"❌ Ошибка при обработке {test_image}: {str(e)}")
    
    print("✅ Тестирование завершено!")
    print(f"Результаты сохранены в директории: {results_dir}")
    return True

def visualize_prediction(image_path, mask_binary, mask_prob, save_path):
    """Создание визуализации предсказания"""
    try:
        # Загружаем исходное изображение
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Создаем фигуру
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Исходное изображение
        axes[0, 0].imshow(image_rgb)
        axes[0, 0].set_title('Исходное изображение')
        axes[0, 0].axis('off')
        
        # Бинарная маска
        axes[0, 1].imshow(mask_binary, cmap='gray')
        axes[0, 1].set_title('Предсказанная маска (бинарная)')
        axes[0, 1].axis('off')
        
        # Вероятностная маска
        axes[1, 0].imshow(mask_prob, cmap='hot', vmin=0, vmax=1)
        axes[1, 0].set_title('Предсказанная маска (вероятности)')
        axes[1, 0].axis('off')
        
        # Наложение маски на изображение
        overlay = image_rgb.copy()
        mask_colored = np.zeros_like(image_rgb)
        mask_colored[:, :, 0] = mask_binary  # Красный канал
        
        result = cv2.addWeighted(image_rgb, 0.7, mask_colored, 0.3, 0)
        axes[1, 1].imshow(result)
        axes[1, 1].set_title('Результат сегментации')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
    except Exception as e:
        print(f"Ошибка при создании визуализации: {str(e)}")

def create_sample_data():
    """Создание примера данных для демонстрации"""
    print("\n" + "="*50)
    print("СОЗДАНИЕ ПРИМЕРА ДАННЫХ")
    print("="*50)
    
    try:
        # Создаем простые синтетические изображения яблок
        def create_apple_image(size=(256, 256), color_variation=0):
            """Создание синтетического изображения яблока"""
            image = np.zeros((*size, 3), dtype=np.uint8)
            
            # Случайные параметры яблока
            center_x = np.random.randint(size[1]//4, 3*size[1]//4)
            center_y = np.random.randint(size[0]//4, 3*size[0]//4)
            radius_x = np.random.randint(40, 80)
            radius_y = np.random.randint(40, 80)
            
            # Базовый цвет яблока (красный с вариациями)
            base_color = [
                max(100, 200 + color_variation),
                max(0, 50 + color_variation//2),
                max(0, 30 + color_variation//3)
            ]
            
            # Рисуем эллипс (яблоко)
            cv2.ellipse(image, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, base_color, -1)
            
            # Добавляем шум и текстуру
            noise = np.random.normal(0, 20, image.shape).astype(np.int16)
            image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            
            # Добавляем фон
            background_color = [
                np.random.randint(200, 255),
                np.random.randint(200, 255), 
                np.random.randint(200, 255)
            ]
            
            mask = np.zeros(size[:2], dtype=np.uint8)
            cv2.ellipse(mask, (center_x, center_y), (radius_x, radius_y), 0, 0, 360, 255, -1)
            
            image[mask == 0] = background_color
            
            return image, mask
        
        # Создаем несколько примеров
        images_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\dataset\images'
        test_dir = r'C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\test'
        
        # Создаем тренировочные данные
        for i in range(5):
            image, _ = create_apple_image(color_variation=np.random.randint(-50, 50))
            
            # Сохраняем изображение
            image_path = os.path.join(images_dir, f'apple{i+1}.jpeg')
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            print(f"Создано тренировочное изображение: apple{i+1}.jpeg")
        
        # Создаем тестовые данные
        for i in range(2):
            image, _ = create_apple_image(color_variation=np.random.randint(-30, 30))
            
            # Сохраняем тестовое изображение
            test_path = os.path.join(test_dir, f'test_apple{i+1}.jpeg')
            cv2.imwrite(test_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            print(f"Создано тестовое изображение: test_apple{i+1}.jpeg")
        
        print("✅ Примеры данных созданы!")
        return True
        
    except Exception as e:
        print(f"❌ Ошибка при создании примеров данных: {str(e)}")
        return False

def load_trained_model(model_path, model_type='unet'):
    """Загрузка обученной модели"""
    try:
        trainer = SegmentationTrainer(model_type=model_type, device='auto')
        trainer.load_model(model_path)
        print(f"✅ Модель загружена из {model_path}")
        return trainer
    except Exception as e:
        print(f"❌ Ошибка при загрузке модели: {str(e)}")
        return None

def main():
    """Главная функция программы"""
    parser = argparse.ArgumentParser(description='Система сегментации объектов')
    parser.add_argument('--mode', type=str, default='train',
                      choices=['train', 'predict'],
                      help='Режим работы: train (обучение) или predict (предсказание)')
    parser.add_argument('--model', type=str, default='unet',
                      choices=['unet', 'simple'],
                      help='Тип модели: unet или simple')
    parser.add_argument('--model_path', type=str,
                      help='Путь к сохраненной модели (для режима predict)')
    parser.add_argument('--test_image', type=str,
                      help='Путь к тестовому изображению (для режима predict)')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Существующий код для обучения
        setup_directories()
        if check_dataset():
            if step1_generate_masks():
                if step2_augment_data():
                    success, trainer = step3_train_model(model_type=args.model)
                    if success:
                        step4_test_model(trainer)
    
    elif args.mode == 'predict':
        if not args.model_path or not args.test_image:
            print("❌ Для режима predict необходимо указать --model_path и --test_image")
            return
        
        # Создаем тренер
        trainer = SegmentationTrainer(model_type=args.model)
        
        # Загружаем модель
        trainer.load_model(args.model_path)
        
        # Получаем предсказание
        mask_binary, mask_prob = trainer.predict(args.test_image)
        
        # Создаем визуализацию
        result_path = r"C:\Users\DiVaN\Documents\DZ_VUZ\AI\vvedeniye\laba3\result\prediction_visualization.png"
        visualize_prediction(args.test_image, mask_binary, mask_prob, result_path)
        
        print(f"✅ Предсказание выполнено и сохранено в {result_path}")

if __name__ == "__main__":
    main()