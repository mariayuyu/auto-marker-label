import csv
import matplotlib.pyplot as plt
import os
import glob

# --------------------------- plot_results.py ----------------------------
# Used for
# * plotting a single run's validation, training loss and accuracy over epochs 
# * plotting bar graph of several different runs and their best validation loss, accuracy 
# ------------------------------------------------------------------------


# Paths
csv_file = 'grid_results/run_12/metrics.csv' 
grid_dir = 'grid_results'
top_n = 12      # top n runs plotted

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
                key = key.strip().lower().replace(" ", "_")  # Normalize key
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
    plt.show()

    # --- Plot Accuracy ---
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, val_accuracies, label='Validation Accuracy', color='green', marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Validation Accuracy Over Epochs')
    plt.grid(True)
    plt.legend()
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


def compare_runs(grid_dir='grid_results', top_n=6):
    run_summaries = []

    for csv_path in glob.glob(os.path.join(grid_dir, 'run_*/metrics.csv')):
        try:
            epochs, train_losses, val_losses, val_accuracies, params = load_training_metrics(csv_path)
            if not val_losses:
                continue

            min_val_loss = min(val_losses)
            best_index = val_losses.index(min_val_loss)
            best_accuracy = val_accuracies[best_index] * 100

            run_name = os.path.basename(os.path.dirname(csv_path))
            run_num = int(run_name.split('_')[1])

            run_summaries.append({
                'run_num': run_num,
                'label': f"{run_name}: LSTM={params.get('lstm_cells', '?')}, lr={params.get('learning_rate', '?')}, bs={params.get('batch_size', '?')}",
                'val_loss': min_val_loss,
                'accuracy': best_accuracy,
                'csv_path': csv_path
            })
        except Exception as e:
            print(f"Failed to read {csv_path}: {e}")

    # Sort and select top runs
    run_summaries.sort(key=lambda x: x['val_loss'])
    run_summaries = run_summaries[:top_n]

    # Prepare values
    run_labels = [f"Run {s['run_num']}" for s in run_summaries]
    val_losses = [s['val_loss'] for s in run_summaries]
    accuracies = [s['accuracy'] for s in run_summaries]
    legends = [s['label'] for s in run_summaries]

    colors = plt.cm.tab10.colors

    # --- Plot 1: Validation Loss ---
    plt.figure(figsize=(10, 5))
    bars = []
    for i, (label, loss) in enumerate(zip(run_labels, val_losses)):
        bar = plt.bar(label, loss, color=colors[i % len(colors)], label=legends[i])
        bars.append(bar)

    plt.ylabel('Validation Loss')
    plt.title('Best Validation Loss per Run')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Run Details', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0.3, 0.6)
    plt.tight_layout()
    plt.show()

    # --- Plot 2: Accuracy ---
    plt.figure(figsize=(10, 5))
    for i, (label, acc) in enumerate(zip(run_labels, accuracies)):
        plt.bar(label, acc, color=colors[i % len(colors)], label=legends[i])

    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy at Best Validation Loss per Run')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Run Details', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(85, 92)
    plt.tight_layout()
    plt.show()


# Main
if __name__ == '__main__':
    #plot_metrics(csv_file, save=True)
    compare_runs(grid_dir, top_n=12)
