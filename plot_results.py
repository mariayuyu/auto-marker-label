import csv
import matplotlib.pyplot as plt
import numpy as np
import os
import glob

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
                'label': f"{run_name}: Dropout={params.get('lstm_dropout', '?')}, momentum={params.get('momentum', '?')}",
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

def plot_loss_vs_accuracy_across_runs(grid_dir='grid_results'):

    run_data = []

    for csv_path in glob.glob(os.path.join(grid_dir, 'run_*/metrics.csv')):
        try:
            epochs, train_losses, val_losses, val_accuracies, params = load_training_metrics(csv_path)
            if not val_losses:
                continue

            min_val_loss = min(val_losses)
            best_index = val_losses.index(min_val_loss)
            best_accuracy = val_accuracies[best_index] * 100

            run_folder = os.path.basename(os.path.dirname(csv_path))
            run_name = run_folder.replace('run_', '')  # Only the number

            run_data.append({
                'loss': min_val_loss,
                'accuracy': best_accuracy,
                #'label': f"Run {run_name} (LSTM cells={params.get('lstm_cells')}, lr={params.get('learning_rate')}, bs={params.get('batch_size')})",
                'label': f"Run {run_name} (dropout={params.get('lstm_dropout')}, momentum={params.get('momentum')})",
            })
        except Exception as e:
            print(f"Failed to process {csv_path}: {e}")

    if not run_data:
        print("No runs found.")
        return

    # --- Plot cluster ---
    plt.figure(figsize=(12, 6))  # slightly wider to fit legend
    for run in run_data:
        plt.scatter(run['loss'], run['accuracy'], label=run['label'], s=80)

    plt.xlabel('Validation Loss', fontsize=14)
    plt.ylabel('Validation Accuracy (%)', fontsize=14)
    plt.title('Validation Accuracy vs. Loss Across Runs', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Annotate with run number
    for run in run_data:
        short_label = run['label'].split(' ')[1]  # Just the number
        plt.annotate(short_label,
                     (run['loss'], run['accuracy']),
                     textcoords="offset points",
                     xytext=(0, 5),
                     ha='center',
                     fontsize=10,
                     #color='red'
                     )

    # Legend to the right of the plot
    plt.legend(title='Run Details',
               bbox_to_anchor=(1.05, 1),
               loc='upper left',
               fontsize=10,
               title_fontsize=12)

    plt.tight_layout()
    plt.show()

def plot_hyperparam_heatmap(grid_dir='grid_results',
                             param_x='batch_size',
                             param_y='lstm_cell_count',
                             metric='accuracy',
                             cmap='viridis'):
   
    data = []
    x_vals_set = set()
    y_vals_set = set()

    for csv_path in glob.glob(os.path.join(grid_dir, 'run_*/metrics.csv')):
        try:
            _, _, val_losses, val_accuracies, params = load_training_metrics(csv_path)
            if not val_losses:
                continue

            best_idx = np.argmin(val_losses)
            score = val_accuracies[best_idx] * 100 if metric == 'accuracy' else val_losses[best_idx]

            x_val = params.get(param_x)
            y_val = params.get(param_y)
            print(f"Params: {params}")
            if x_val is not None and y_val is not None:
                data.append((x_val, y_val, score))
                x_vals_set.add(x_val)
                y_vals_set.add(y_val)
        except Exception as e:
            print(f"Failed to process {csv_path}: {e}")

    if not data:
        print("No matching data found.")
        return

    # Sort axis values
    x_vals = sorted(x_vals_set)
    y_vals = sorted(y_vals_set)

    # Create lookup grid
    heatmap = np.full((len(y_vals), len(x_vals)), np.nan)

    for x, y, score in data:
        xi = x_vals.index(x)
        yi = y_vals.index(y)
        heatmap[yi, xi] = score

    # Plot heatmap
    plt.figure(figsize=(8, 6))
    im = plt.imshow(heatmap, cmap=cmap, aspect='auto', origin='lower')

    plt.xticks(ticks=np.arange(len(x_vals)), labels=x_vals, fontsize=12)
    plt.yticks(ticks=np.arange(len(y_vals)), labels=y_vals, fontsize=12)
    plt.xlabel(param_x.replace('_', ' ').title(), fontsize=14)
    plt.ylabel(param_y.replace('_', ' ').title(), fontsize=14)
    plt.title(f"Grid Search: {metric.title()} Heatmap", fontsize=16)

    # Add value labels
    for i in range(len(y_vals)):
        for j in range(len(x_vals)):
            val = heatmap[i, j]
            if not np.isnan(val):
                text = f"{val:.2f}" if metric == 'accuracy' else f"{val:.3f}"
                plt.text(j, i, text, ha='center', va='center', color='white' if val < np.nanmax(heatmap)*0.5 else 'black')

    plt.colorbar(im, label=metric.title())
    plt.tight_layout()
    plt.show()

# Main
if __name__ == '__main__':
    csv_file = 'grid_results_2/run_5/metrics.csv'  
    csv_file2 = 'metrics_20250704_170540.csv'
    #plot_metrics(csv_file, save=True)
    #compare_runs(grid_dir='grid_results_2', top_n=9)
    #plot_loss_vs_accuracy_across_runs(grid_dir='grid_results_2')

    # Grid search 1: batch size vs lstm cell count, show accuracy
    plot_hyperparam_heatmap(grid_dir='grid_results',
                            param_x='batch_size',
                            #param_x='learning_rate',
                            #param_y='lstm_cells',
                            param_y='learning_rate',
                            metric='accuracy',
                            cmap="Blues")

    # Grid search 2: dropout vs momentum, show validation loss
    plot_hyperparam_heatmap(grid_dir='grid_results_2',
                            param_x='lstm_dropout',
                            param_y='momentum',
                            metric='accuracy',
                            cmap="Blues"
                            )
