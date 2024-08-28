
import torch
from info_theory_experiments.models import SkipConnectionSupervenientFeatureNetwork
import itertools

# Load the model
model_path = "models/NEURIPS-other-gammas-for-bits-0.9-sleek-durian-2.pth"
model = SkipConnectionSupervenientFeatureNetwork(num_atoms=6, feature_size=1, hidden_sizes=[256,256])
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

# Generate all possible 6-bit combinations
all_combinations = list(itertools.product([0, 1], repeat=6))
print(all_combinations)

inputs = torch.tensor(all_combinations, dtype=torch.float32)

# Pass all combinations through the model
with torch.no_grad():
    outputs = model(inputs)

    # Convert outputs to a 1D numpy array for plotting
    outputs_np = outputs.numpy().flatten()

    # Import matplotlib for plotting
    import matplotlib.pyplot as plt

    # Create a line plot
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(outputs_np)), outputs_np, '-o')
    plt.title('Model Outputs for All 6-bit Combinations')
    plt.xlabel('Input Combination Index')
    plt.ylabel('Model Output')
    plt.grid(True)

    # Save the plot
    plt.savefig('figures/model_outputs_plot.png')
    plt.close()

    print(f"Plot saved as 'model_outputs_plot.png'")
