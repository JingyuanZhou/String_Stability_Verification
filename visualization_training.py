from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(8, 6))
vis_index = [str(i) for i in range(8)]

num_colors = len(vis_index)
color_map = plt.get_cmap("tab10", num_colors)  # Dynamically get 'tab10' colormap
color_index = color_map(np.arange(num_colors))  # Assign unique colors
for index in range(len(vis_index)):
    log_dir = "lightning_logs/version_" + vis_index[index]
    reader = SummaryReader(log_dir)

    df = reader.scalars
    # Filter train_loss data
    train_loss_df = df[df["tag"] == "train_loss"]
    train_loss_epoch_df = df[df["tag"] == "train_loss_epoch"]

    # Plot train loss
    plt.plot(train_loss_df["step"], train_loss_df["value"], marker='o', linestyle='-', 
            color=color_index[index], label="Iter "+str(index)) # changing different color for each line

    plt.legend()

plt.xlabel("Training Step")
plt.ylabel("Loss Value")
plt.title("Train Loss Over Steps")
plt.grid(True)
plt.show()

# print all the tags
print(reader.tags)
