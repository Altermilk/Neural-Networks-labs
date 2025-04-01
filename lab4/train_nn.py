import time
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import lr_scheduler

from dataset import train_loader, test_loader, species  # Импорт датасета
 
class NeuralNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Сверточные слои
        self.conv1 = nn.Conv2d(3, 32, 3)  # (32, 30, 30)
        self.conv2 = nn.Conv2d(32, 64, 3) # (64, 28, 28)
        self.conv3 = nn.Conv2d(64, 128, 3) # (128, 26, 26)

        # Слой пулинга
        self.pool = nn.MaxPool2d(2, 2)

        # Пакетная нормализация
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        # Вычисляем размер входа в полносвязанные слои
        self.flatten_size = self._get_flatten_size()

        # Полносвязанные слои
        self.fc1 = nn.Linear(self.flatten_size, 512)  
        self.fc2 = nn.Linear(512, len(species))  

    def _get_flatten_size(self):
        """Функция для вычисления размера входа в fc1."""
        x = torch.randn(1, 3, 32, 32)  
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        return x.numel()  

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
    
def count_acc(net):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

if __name__ == '__main__':

    net = NeuralNet()
    epoches = 50

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=0.01, weight_decay=5e-4)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoches, eta_min=0.001)

    train_losses = []
    accuracies = []

    # Early Stopping параметры
    patience = 5  # Количество эпох без улучшения, после которого остановится обучение
    best_accuracy = 0
    counter = 0

    start_time = time.time()

    for epoch in range(epoches):
        print(f'Epoch {epoch+1}/{epoches}...')
        net.train()
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), desc="     train progress", unit="IMG", total=len(train_loader)):
            inputs, labels = data

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        train_losses.append(avg_loss)
        print(f'  -> Loss: {avg_loss:4f}')

        # Подсчет accuracy
        accuracy = count_acc(net)
        accuracies.append(accuracy)

        print(f'  -> Accuracy: {accuracy:.2f}%')

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            counter = 0  # Сбрасываем счетчик
            torch.save(net.state_dict(), 'best_model.pth')  # Сохраняем лучшую модель
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs!")
                break  # Останавливаем обучение

        scheduler.step()

    spent_time = time.time() - start_time
    minutes = int(spent_time // 60)
    seconds = int(spent_time % 60)

    print(f'  -> Time taken: {minutes} min {seconds} sec\n')
    torch.save(net.state_dict(), 'trained_net3.pth')

    from graph import *
    build_graph(len(accuracies), train_losses, accuracies)