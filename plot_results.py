import csv
import matplotlib.pyplot as plt

def load_training_metrics(csv_path):
    epochs = []
    train_losses = []
    val_losses = []
    val_accuracies = []
    params = {}

    with open(csv_path, 'r') as file:
        reader = csv.reader(file)
        lines = list(reader)

        # Extract parameters from metadata
        i = 0
        while i < len(lines):
            row = lines[i]
            if not row or row[0].startswith('Epoch'):
                break
            if len(row) == 2 and not row[0].startswith("#"):
                key, val = row
                try:
                    params[key] = float(val)
                except ValueError:
                    params[key] = val
            i += 1

        # Extract data rows
        header_index = next(idx for idx, row in enumerate(lines) if row and row[0] == "Epoch")
        for row in lines[header_index + 1:]:
            if len(row) >= 4:
                epoch = int(row[0])
                train_loss = float(row[1])
                val_loss = float(row[2])
                val_acc = float(row[3])
                epochs.append(epoch)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc)

    return epochs, train_losses, val_losses, val_accuracies, params


def plot_metrics(csv_path, save=False, outdir='plots'):
    import os
    if save and not os.path.exists(outdir):
        os.makedirs(outdir)

    epochs, train_losses, val_losses, val_accuracies, params = load_training_metrics(csv_path)

    # --- Plot Losses ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, train_losses, label='Train Loss', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(os.path.join(outdir, 'loss_plot.png'))
    plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
    if save:
        plt.savefig(os.path.join(outdir, 'accuracy_plot.png'))
    plt.show()

    # --- Print Parameters ---
    print("\nTraining Parameters:")
    for key, val in params.items():
        print(f"  {key}: {val}")

    # --- Best Validation Loss Info ---
    min_val_loss = min(val_losses)
    min_index = val_losses.index(min_val_loss)
    best_epoch = epochs[min_index]
    best_accuracy = val_accuracies[min_index] * 100

    print(f"\nBest Model:")
    print(f"  Epoch: {best_epoch}")
    print(f"  Validation Loss: {min_val_loss:.4f}")
    print(f"  Corresponding Accuracy: {best_accuracy:.2f}%")


# Main
if __name__ == '__main__':
    csv_file = 'metrics.csv'  
    plot_metrics(csv_file, save=True)
