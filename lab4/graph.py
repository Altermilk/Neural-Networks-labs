import matplotlib.pyplot as plt

def build_graph(epochs, train_losses, accuracies):
    fig, ax1 = plt.subplots(figsize=(10, 5))

    epochs_range = range(1, epochs + 1)

    # Линия для loss (левая ось Y)
    ax1.plot(epochs_range, train_losses, marker="o", linestyle="-", label="Training Loss", color="red")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss", color="red")
    ax1.tick_params(axis="y", labelcolor="red")

    # Добавляем подписи значений Loss
    for x, y in zip(epochs_range, train_losses):
        ax1.text(x, y, f"{y:.2f}", ha="right", va="bottom", fontsize=10, color="red")

    # Вторая ось Y для accuracy
    ax2 = ax1.twinx()
    ax2.plot(epochs_range, accuracies, marker="s", linestyle="-", label="Accuracy", color="blue")
    ax2.set_ylabel("Accuracy (%)", color="blue")
    ax2.tick_params(axis="y", labelcolor="blue")

    # Добавляем подписи значений Accuracy
    for x, y in zip(epochs_range, accuracies):
        ax2.text(x, y, f"{y:.2f}%", ha="left", va="bottom", fontsize=10, color="blue")

    # Заголовок
    plt.title("Training Loss and Accuracy Over Epochs")

    # Легенды
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # Показываем график
    plt.show()