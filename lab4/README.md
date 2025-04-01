# Author
Kalinich A.B., KC-44
# Link to kaggle competition
https://www.kaggle.com/datasets/phucthaiv02/butterfly-image-classification/data
# download datasets
```python
import kagglehub

# Download latest version
path = kagglehub.dataset_download("phucthaiv02/butterfly-image-classification")

print("Path to dataset files:", path)
```
# Model structure
```python
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

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))

        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x)) 
        x = self.fc2(x)
        return x
```
## Tensor transformation filters
```python
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```
## Learning cycle
```python
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

        # Подсчет accuracy
        accuracy = count_acc(net)
        accuracies.append(accuracy)

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
```
