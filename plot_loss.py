import pandas as pd
import matplotlib.pyplot as plt

# Read data from CSV file
csv_file = "losses.csv"  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Extract discriminator and generator losses
discriminator_loss = data['discriminator_loss']
generator_loss = data['generator_loss']

# Create a plot
plt.figure(figsize=(10, 6))
plt.plot(discriminator_loss, label="Discriminator Loss", color="blue")
plt.plot(generator_loss, label="Generator Loss", color="red")

# Add labels and legend
plt.title("Discriminator and Generator Losses")
plt.xlabel("Epoch")
plt.ylabel("Cumulative Loss")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()

