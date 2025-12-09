import matplotlib.pyplot as plt
import pandas as pd
import sys
import os

plt.style.use("seaborn-v0_8-whitegrid")
COLORS = ["#1f77b4", "#ff7f0e"]

if len(sys.argv) < 3:
    print("Uso: python script.py <file.csv> <optimization_level> <string>")
    sys.exit(1)

file_path = sys.argv[1]
optimization_level = sys.argv[2]
extra_string = sys.argv[3]

file_name = os.path.splitext(os.path.basename(file_path))[0]
output_dir = "Graphs"
os.makedirs(output_dir, exist_ok=True)
output_base = f"{file_name}_{extra_string}"

data = pd.read_csv(file_path, sep=';')
versions = data['Version']
elapsed_time = data['Elapsed Time (s)']
speedup = data['Speedup']

fig, ax = plt.subplots(1, 2, figsize=(14,6))

ax[0].bar(versions, elapsed_time, color=COLORS)
ax[0].set_title("Execution Time")
ax[0].set_ylabel("Elapsed Time (s)")
for i, v in enumerate(elapsed_time):
    ax[0].text(i, v * 1.01, f"{v:.2f}", ha='center', fontsize=10)

ax[1].bar(versions, speedup, color=COLORS)
ax[1].set_title("Speedup")
ax[1].set_ylabel("Speedup")
for i, v in enumerate(speedup):
    ax[1].text(i, v * 1.01, f"{v:.2f}", ha='center', fontsize=10)

fig.suptitle(f"Performance Overview ({optimization_level}, {extra_string})", fontsize=14)

plt.subplots_adjust(top=0.85, bottom=0.15)
output_path = os.path.join(output_dir, f"{output_base}.png")
plt.savefig(output_path, dpi=300)
plt.close()


