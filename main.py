import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import os
import gc
import threading
import psutil
import time
import multiprocessing
from collections import defaultdict

# Конфигурация ресурсов системы
class ResourceConfig:
    def __init__(self,
                 max_cpu_usage=80,
                 max_memory_usage=85,
                 batch_size=32,
                 num_workers=2,
                 pin_memory=True,
                 mixed_precision=True,
                 gradient_accumulation=1,
                 checkpoint_freq=5):

        self.max_cpu_usage = max_cpu_usage
        self.max_memory_usage = max_memory_usage
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.mixed_precision = mixed_precision
        self.gradient_accumulation = gradient_accumulation
        self.checkpoint_freq = checkpoint_freq

        self._auto_adjust_parameters()

    def _auto_adjust_parameters(self):
        cpu_count = psutil.cpu_count(logical=False) if psutil.cpu_count(logical=False) else 2
        memory_gb = psutil.virtual_memory().total / (1024 ** 3)

        if os.name == 'nt':
            self.num_workers = 0
        else:
            if self.num_workers == -1:
                self.num_workers = min(cpu_count, 4)

        if torch.cuda.is_available():
            try:
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                if gpu_memory_gb < 4:
                    self.batch_size = min(self.batch_size, 8)
                elif gpu_memory_gb < 8:
                    self.batch_size = min(self.batch_size, 16)
            except Exception:
                self.batch_size = min(self.batch_size, 16)
        else:
            if memory_gb < 8:
                self.batch_size = min(self.batch_size, 16)

        if not torch.cuda.is_available():
            self.mixed_precision = False

        print(f"Авто-настройки:")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Num workers: {self.num_workers}")
        print(f"  Mixed precision: {self.mixed_precision}")


# Мониторинг ресурсов
class ResourceMonitor:
    def __init__(self, config):
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.resource_history = defaultdict(list)

    def start_monitoring(self):
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_monitoring(self):
        self.monitoring = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)

    def _monitor_loop(self):
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=0.1)
                memory_percent = psutil.virtual_memory().percent

                self.resource_history['cpu'].append(cpu_percent)
                self.resource_history['memory'].append(memory_percent)

                if cpu_percent > self.config.max_cpu_usage or memory_percent > self.config.max_memory_usage:
                    print(f"High resources usage: CPU {cpu_percent:.1f}%, RAM {memory_percent:.1f}%")
                    time.sleep(0.5)

                time.sleep(1.0)
            except Exception as e:
                print(f"Resource monitor error: {e}")
                break

    def get_resource_usage(self):
        try:
            gpu_usage = 0
            if torch.cuda.is_available():
                gpu_usage = torch.cuda.memory_allocated() / (1024 ** 3) if torch.cuda.memory_allocated() > 0 else 0

            return {
                'cpu': psutil.cpu_percent(),
                'memory': psutil.virtual_memory().percent,
                'gpu_memory_gb': gpu_usage
            }
        except Exception:
            return {'cpu': 0, 'memory': 0, 'gpu_memory_gb': 0}


# Класс для создания датасета СОБАК из CIFAR-10
class DogDataset(Dataset):
    def __init__(self, cifar_dataset, max_samples=None):
        self.data = []
        self.labels = []
        sample_count = 0
        dog_count = 0
        non_dog_count = 0

        print("Создание датасета собак...")

        for i, (img, label) in enumerate(cifar_dataset):
            if max_samples and sample_count >= max_samples:
                break

            # CIFAR-10: label 5 == dog
            if label == 5 and dog_count < (max_samples // 2 if max_samples else 2500):
                self.data.append(img)
                self.labels.append(1)  # 1 - собака
                dog_count += 1
                sample_count += 1
            elif label != 5 and i % 5 == 0 and non_dog_count < (max_samples // 2 if max_samples else 2500):
                self.data.append(img)
                self.labels.append(0)  # 0 - не собака
                non_dog_count += 1
                sample_count += 1

        print(f"Создан датасет: {dog_count} собак, {non_dog_count} не-собак")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# Оптимизированная архитектура
class OptimizedDogDetectionCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(OptimizedDogDetectionCNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(dropout_rate / 2),

            nn.AdaptiveAvgPool2d((4, 4))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x


def cleanup_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def train_model_optimized(model, train_loader, test_loader, resource_monitor, scaler, scheduler, criterion, optimizer,
                          num_epochs=10, device='cpu'):
    train_losses = []
    train_accuracies = []
    test_accuracies = []

    resource_monitor.start_monitoring()

    best_test_acc = 0.0
    patience = 3
    patience_counter = 0

    try:
        for epoch in range(num_epochs):
            print(f"\nEpoch [{epoch + 1}/{num_epochs}]")

            resources = resource_monitor.get_resource_usage()
            print(f"Resource usage - CPU: {resources['cpu']:.1f}%, RAM: {resources['memory']:.1f}%")
            if torch.cuda.is_available():
                print(f"GPU memory: {resources['gpu_memory_gb']:.2f} GB")

            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                try:
                    data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                    optimizer.zero_grad()

                    if scaler and resource_config.mixed_precision and torch.cuda.is_available():
                        with torch.cuda.amp.autocast():
                            output = model(data)
                            loss = criterion(output, target)
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                    running_loss += loss.item()
                    _, predicted = torch.max(output.data, 1)
                    total_train += target.size(0)
                    correct_train += (predicted == target).sum().item()

                    if batch_idx % 20 == 0:
                        print(f'  Batch [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')

                except Exception as e:
                    print(f"Batch error {batch_idx}: {e}")
                    continue

            train_accuracy = 100 * correct_train / total_train if total_train > 0 else 0
            avg_loss = running_loss / len(train_loader) if len(train_loader) > 0 else 0

            model.eval()
            correct_test = 0
            total_test = 0

            with torch.no_grad():
                for data, target in test_loader:
                    try:
                        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

                        if scaler and resource_config.mixed_precision and torch.cuda.is_available():
                            with torch.cuda.amp.autocast():
                                output = model(data)
                        else:
                            output = model(data)

                        _, predicted = torch.max(output.data, 1)
                        total_test += target.size(0)
                        correct_test += (predicted == target).sum().item()
                    except Exception as e:
                        print(f"Test error: {e}")
                        continue

            test_accuracy = 100 * correct_test / total_test if total_test > 0 else 0

            train_losses.append(avg_loss)
            train_accuracies.append(train_accuracy)
            test_accuracies.append(test_accuracy)

            print(f'  Train Acc: {train_accuracy:.2f}%')
            print(f'  Test Acc: {test_accuracy:.2f}%')
            print(f'  Avg Loss: {avg_loss:.4f}')

            if test_accuracy > best_test_acc:
                best_test_acc = test_accuracy
                patience_counter = 0
                try:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_acc': best_test_acc,
                    }, 'best_dog_detection_model.pth')
                except Exception as e:
                    print(f"Save model error: {e}")
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

            if scheduler:
                scheduler.step()

            cleanup_memory()
            print('-' * 50)

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Training error: {e}")
    finally:
        resource_monitor.stop_monitoring()
        cleanup_memory()

    return train_losses, train_accuracies, test_accuracies, best_test_acc


def load_image_from_url(url, timeout=10):
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        return img
    except Exception as e:
        print(f"Error loading image from {url}: {e}")
        return None


def predict_dog_optimized(model, image_path_or_url, transform, device='cpu'):
    try:
        model.eval()

        if isinstance(image_path_or_url, str) and image_path_or_url.startswith('http'):
            image = load_image_from_url(image_path_or_url)
        else:
            image = Image.open(image_path_or_url).convert('RGB')

        if image is None:
            return None, None, None, None

        input_tensor = transform(image).unsqueeze(0).to(device, non_blocking=True)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            confidence_scores = probabilities.cpu().numpy()[0]
            predicted = torch.argmax(probabilities, dim=1).item()

        is_dog = predicted == 1
        confidence = confidence_scores[predicted] * 100

        dog_confidence = confidence_scores[1] * 100
        not_dog_confidence = confidence_scores[0] * 100

        return is_dog, confidence, image, {
            'dog_probability': dog_confidence,
            'not_dog_probability': not_dog_confidence
        }
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, None, None, None


def main():
    try:
        torch.manual_seed(42)
        np.random.seed(42)

        global resource_config
        resource_config = ResourceConfig(
            max_cpu_usage=70,
            max_memory_usage=80,
            batch_size=16,
            num_workers=0 if os.name == 'nt' else 2,
            mixed_precision=torch.cuda.is_available(),
            gradient_accumulation=1
        )

        resource_monitor = ResourceMonitor(resource_config)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {device}')

        transform_train = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.RandomHorizontalFlip(p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        print("Downloading CIFAR-10 dataset...")
        try:
            cifar_train = CIFAR10(root='./data', train=True, transform=transform_train, download=True)
            cifar_test = CIFAR10(root='./data', train=False, transform=transform_test, download=True)
        except Exception as e:
            print(f"Error downloading CIFAR-10: {e}")
            return

        train_dataset = DogDataset(cifar_train, max_samples=1000)
        test_dataset = DogDataset(cifar_test, max_samples=200)

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Test dataset size: {len(test_dataset)}")

        train_loader = DataLoader(
            train_dataset,
            batch_size=resource_config.batch_size,
            shuffle=True,
            num_workers=resource_config.num_workers,
            pin_memory=resource_config.pin_memory and torch.cuda.is_available(),
            drop_last=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=resource_config.batch_size,
            shuffle=False,
            num_workers=resource_config.num_workers,
            pin_memory=resource_config.pin_memory and torch.cuda.is_available(),
            drop_last=False
        )

        model = OptimizedDogDetectionCNN(dropout_rate=0.2).to(device)
        total_params = sum(p.numel() for p in model.parameters())

        print(f"Model created:")
        print(f"  Total params: {total_params:,}")

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

        scaler = torch.cuda.amp.GradScaler() if resource_config.mixed_precision and torch.cuda.is_available() else None

        print("\nStart training...")
        print(f"  Batch size: {resource_config.batch_size}")
        print(f"  Mixed precision: {resource_config.mixed_precision}")
        print("-" * 50)

        train_losses, train_accs, test_accs, best_acc = train_model_optimized(
            model, train_loader, test_loader, resource_monitor, scaler, scheduler,
            criterion, optimizer, num_epochs=8, device=device
        )

        print("\nTraining finished!")
        if test_accs:
            print(f"Final test acc: {test_accs[-1]:.2f}%")
            print(f"Best test acc: {best_acc:.2f}%")

        if train_losses and train_accs and test_accs:
            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.plot(train_losses)
            plt.title('Training Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.grid(True)

            plt.subplot(1, 3, 2)
            plt.plot(train_accs, label='Train Acc', marker='o')
            plt.plot(test_accs, label='Test Acc', marker='s')
            plt.title('Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy (%)')
            plt.legend()
            plt.grid(True)

            plt.subplot(1, 3, 3)
            model_info = [
                f"Params: {total_params:,}",
                f"Batch size: {resource_config.batch_size}",
                f"Best acc: {best_acc:.2f}%",
                f"Device: {device}"
            ]
            plt.text(0.1, 0.5, '\n'.join(model_info), fontsize=12,
                     verticalalignment='center', transform=plt.gca().transAxes)
            plt.title('Model Info')
            plt.axis('off')

            plt.tight_layout()
            plt.show()

        cleanup_memory()
        print("Done!")

    except KeyboardInterrupt:
        print("Program interrupted by user")
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cleanup_memory()


if __name__ == '__main__':
    if hasattr(multiprocessing, 'set_start_method'):
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            pass

    multiprocessing.freeze_support()
    main()
