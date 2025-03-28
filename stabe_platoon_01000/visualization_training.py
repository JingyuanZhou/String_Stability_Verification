from tbparse import SummaryReader
import matplotlib.pyplot as plt
import numpy as np

num_ce_list = np.load("data/num_ce_list.npy")
num_veri_time_list = np.load("data/num_veri_time_list.npy")

print(num_ce_list)
print(num_veri_time_list)

#plt.style.use('seaborn-white')  # 使用清爽的背景样式
# Set font to Times New Roman
plt.rcParams["font.family"] = "Times New Roman"

# Increase figure resolution
plt.figure(figsize=(8, 6), dpi=300)  # Higher DPI for better clarity

vis_index = [str(958+i) for i in range(1)]
num_colors = len(vis_index)
color_map = plt.get_cmap("tab10", num_colors)  # Dynamically get 'tab10' colormap
color_index = color_map(np.arange(num_colors))  # Assign unique colors

for index in range(len(vis_index)):
    log_dir = "lightning_logs/version_" + vis_index[index]
    reader = SummaryReader(log_dir)

    df = reader.scalars
    # Filter train_loss data
    df = df[df["tag"] == "val_loss"]

    epoch_df = df[df["tag"] == "train_loss_epoch"]


    # Plot train loss
    plt.plot(df["step"], df["value"], marker='o', linestyle='-',
             color=color_index[index], label="Iter " + str(index), markersize = 1.5)  # Assign different colors

# Update legend with two columns
plt.legend(ncol=2, fontsize=10, loc="upper left")

# Set axis labels
plt.xlabel("Training Step", fontsize=14)
plt.ylabel("Loss Value", fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Increase grid clarity
plt.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

# Save figure as PDF with high resolution
plt.savefig("output_figures/train_loss_plot.pdf", format="pdf", dpi=300, bbox_inches="tight")

# Show plot
plt.show()

