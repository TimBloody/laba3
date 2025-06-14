import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, jaccard_score
import matplotlib.pyplot as plt
from tqdm import tqdm

class SegmentationDataset(Dataset):
    """Датасет для семантической сегментации"""
    
    def __init__(self, images_dir, masks_dir, img_size=(256, 256), transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.img_size = img_size
        self.transform = transform
        
        # Получаем список всех изображений
        self.image_files = [f for f in os.listdir(images_dir) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        # Фильтруем только те, для которых есть маски
        self.valid_files = []
        for img_file in self.image_files:
            mask_file = self._get_mask_filename(img_file)
            if mask_file is not None:
                mask_path = os.path.join(masks_dir, mask_file)
                if os.path.exists(mask_path):
                    self.valid_files.append(img_file)
        
        if not self.valid_files:
            raise ValueError("Не найдено ни одной пары изображение-маска!")
        
        print(f"Найдено {len(self.valid_files)} пар изображение-маска")
    
    def _get_mask_filename(self, img_filename):
        """Получение имени файла маски по имени изображения"""
        base_name = os.path.splitext(img_filename)[0]
        # Ищем соответствующую маску в директории
        mask_files = [f for f in os.listdir(self.masks_dir) 
                     if f.lower().endswith('.png')]
        
        # Если это оригинальное изображение
        if base_name.startswith('orig_'):
            # Ищем маску с тем же именем
            for mask_file in mask_files:
                if mask_file.startswith('orig_'):
                    return mask_file
        
        # Если это аугментированное изображение
        # Ищем маску с тем же суффиксом аугментации
        for mask_file in mask_files:
            if any(suffix in mask_file for suffix in ['_rot_', '_flip_', '_bright_', '_dark_', '_blur_', '_noise_', '_albu_']):
                # Извлекаем суффикс аугментации из имени изображения
                aug_suffix = '_'.join(base_name.split('_')[-2:])  # берем последние две части имени
                if aug_suffix in mask_file:
                    return mask_file
        
        # Если маска не найдена, возвращаем None
        return None
    
    def __len__(self):
        return len(self.valid_files)
    
    def __getitem__(self, idx):
        # Загружаем изображение
        img_path = os.path.join(self.images_dir, self.valid_files[idx])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Загружаем маску
        mask_file = self._get_mask_filename(self.valid_files[idx])
        mask_path = os.path.join(self.masks_dir, mask_file)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Изменяем размер
        image = cv2.resize(image, self.img_size)
        mask = cv2.resize(mask, self.img_size)
        
        # Нормализуем изображение
        image = image.astype(np.float32) / 255.0
        
        # Бинаризуем маску (0 или 1)
        mask = (mask > 127).astype(np.float32)
        
        # Преобразуем в тензоры PyTorch
        image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0)  # HW -> 1HW
        
        return image, mask

class UNet(nn.Module):
    """
    U-Net архитектура для семантической сегментации
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(UNet, self).__init__()
        
        # Encoder (сжимающий путь)
        self.enc1 = self.double_conv(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = self.double_conv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.enc3 = self.double_conv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        
        self.enc4 = self.double_conv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        
        # Bottleneck
        self.bottleneck = self.double_conv(512, 1024)
        
        # Decoder (расширяющий путь)
        self.upconv4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.dec4 = self.double_conv(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = self.double_conv(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = self.double_conv(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = self.double_conv(128, 64)
        
        # Выходной слой
        self.final_conv = nn.Conv2d(64, out_channels, 1)
        
    def double_conv(self, in_channels, out_channels):
        """Двойная свертка с BatchNorm и ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.dec1(dec1)
        
        # Выход
        output = self.final_conv(dec1)
        return torch.sigmoid(output)

class SimpleSegmentationCNN(nn.Module):
    """
    Простая CNN для сегментации (альтернатива U-Net)
    """
    
    def __init__(self, in_channels=3, out_channels=1):
        super(SimpleSegmentationCNN, self).__init__()
        
        # Encoder
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_dec3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_dec2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_dec1 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(16, out_channels, 1)
        self.pool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.pool(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.pool(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.pool(conv3)
        
        # Decoder
        up3 = self.upconv3(pool3)
        dec3 = self.conv_dec3(up3)
        
        up2 = self.upconv2(dec3)
        dec2 = self.conv_dec2(up2)
        
        up1 = self.upconv1(dec2)
        dec1 = self.conv_dec1(up1)
        
        output = self.final_conv(dec1)
        return torch.sigmoid(output)

class SegmentationTrainer:
    """
    Класс для обучения модели сегментации
    """
    
    def __init__(self, model_type='unet', device='auto'):
        self.device = self._get_device(device)
        self.model_type = model_type
        
        # Создаем модель
        if model_type == 'unet':
            self.model = UNet(in_channels=3, out_channels=1).to(self.device)
        else:
            self.model = SimpleSegmentationCNN(in_channels=3, out_channels=1).to(self.device)
        
        # История обучения
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def _get_device(self, device):
        """Определение устройства для вычислений"""
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def dice_loss(self, pred, target, smooth=1):
        """Dice Loss для сегментации"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)
        
        return 1 - dice
    
    def combined_loss(self, pred, target):
        """Комбинированная функция потерь (BCE + Dice)"""
        bce = F.binary_cross_entropy(pred, target)
        dice = self.dice_loss(pred, target)
        return 0.5 * bce + 0.5 * dice
    
    def calculate_iou(self, pred, target, threshold=0.5):
        """Вычисление IoU (Intersection over Union)"""
        pred_binary = (pred > threshold).float()
        target_binary = target.float()
        
        intersection = (pred_binary * target_binary).sum()
        union = pred_binary.sum() + target_binary.sum() - intersection
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return (intersection / union).item()
    
    def train_epoch(self, dataloader, optimizer, criterion):
        """Обучение одной эпохи"""
        self.model.train()
        running_loss = 0.0
        running_iou = 0.0
        
        for images, masks in tqdm(dataloader, desc="Training"):
            images, masks = images.to(self.device), masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            running_iou += self.calculate_iou(outputs, masks)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        
        return epoch_loss, epoch_iou
    
    def validate_epoch(self, dataloader, criterion):
        """Валидация одной эпохи"""
        self.model.eval()
        running_loss = 0.0
        running_iou = 0.0
        
        with torch.no_grad():
            for images, masks in tqdm(dataloader, desc="Validation"):
                images, masks = images.to(self.device), masks.to(self.device)
                
                outputs = self.model(images)
                loss = criterion(outputs, masks)
                
                running_loss += loss.item()
                running_iou += self.calculate_iou(outputs, masks)
        
        epoch_loss = running_loss / len(dataloader)
        epoch_iou = running_iou / len(dataloader)
        
        return epoch_loss, epoch_iou
    
    def train(self, train_loader, val_loader, epochs=50, lr=0.001):
        """Основной цикл обучения"""
        print(f"Обучение модели {self.model_type} на устройстве: {self.device}")
        print(f"Параметров в модели: {sum(p.numel() for p in self.model.parameters())}")
        
        # Оптимизатор и функция потерь
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        criterion = self.combined_loss
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 10
        
        for epoch in range(epochs):
            print(f"\nЭпоха {epoch + 1}/{epochs}")
            
            # Обучение
            train_loss, train_iou = self.train_epoch(train_loader, optimizer, criterion)
            
            # Валидация
            val_loss, val_iou = self.validate_epoch(val_loader, criterion)
            
            # Обновляем планировщик
            scheduler.step(val_loss)
            
            # Сохраняем историю
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_iou)
            self.val_accuracies.append(val_iou)
            
            print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
            print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Сохраняем лучшую модель
                torch.save(self.model.state_dict(), f'best_{self.model_type}_model.pth')
                print("Сохранена лучшая модель")
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    print(f"Early stopping после {epoch + 1} эпох")
                    break
        
        print("Обучение завершено!")
        return self.model
    
    def plot_training_history(self):
        """Построение графиков обучения"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # График потерь
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Loss во время обучения')
        ax1.set_xlabel('Эпоха')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # График IoU
        ax2.plot(self.train_accuracies, label='Train IoU')
        ax2.plot(self.val_accuracies, label='Validation IoU')
        ax2.set_title('IoU во время обучения')
        ax2.set_xlabel('Эпоха')
        ax2.set_ylabel('IoU')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def predict(self, image_path, save_path=None):
        """Предсказание маски для одного изображения"""
        self.model.eval()
        
        # Загружаем и предобрабатываем изображение
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_size = image_rgb.shape[:2]
        
        # Изменяем размер и нормализуем
        image_resized = cv2.resize(image_rgb, (256, 256))
        image_tensor = torch.from_numpy(image_resized.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        image_tensor = image_tensor.to(self.device)
        
        # Предсказание
        with torch.no_grad():
            output = self.model(image_tensor)
            mask_pred = output.squeeze().cpu().numpy()
        
        # Возвращаем к исходному размеру
        mask_pred = cv2.resize(mask_pred, (original_size[1], original_size[0]))
        mask_binary = (mask_pred > 0.5).astype(np.uint8) * 255
        
        if save_path:
            cv2.imwrite(save_path, mask_binary)
        
        return mask_binary, mask_pred
    
    def save_model(self, path):
        """Сохранение модели"""
        # Сохраняем только состояние модели
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        """Загрузка модели"""
        # Загружаем состояние модели
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()  # Переводим модель в режим оценки

def create_data_loaders(images_dir, masks_dir, batch_size=8, val_split=0.2, img_size=(256, 256)):
    """Создание data loaders для обучения и валидации"""
    
    # Создаем датасет
    dataset = SegmentationDataset(images_dir, masks_dir, img_size=img_size)
    
    if len(dataset) == 0:
        raise ValueError("Датасет пуст! Проверьте пути к изображениям и маскам")
    
    # Разделяем на обучающую и валидационную выборки
    train_size = int((1 - val_split) * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Создаем data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    print(f"Создан датасет: {len(dataset)} изображений")
    print(f"Обучающая выборка: {train_size} изображений")
    print(f"Валидационная выборка: {val_size} изображений")
    
    return train_loader, val_loader