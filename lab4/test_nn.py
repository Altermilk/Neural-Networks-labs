from train_nn import NeuralNet, test_loader
import torch
from tqdm import tqdm

def test_model():
    print('-- TESTING --')

    path = 'best_model.pth'
    # path = 'trained_net3.pth'
    net = NeuralNet()
    net.load_state_dict(torch.load(path))
    net.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Testing Progress", unit="batch"):
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
    print(f'Accuracy: {100 * correct / total:.2f}%')

if __name__ == '__main__':
    test_model()
