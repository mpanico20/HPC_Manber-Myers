import matplotlib.pyplot as plt
import pandas as pd
import sys

# Controllo se è stato passato il percorso del file
if len(sys.argv) < 2:
    print("Uso: python script.py percorso_del_file.csv")
    sys.exit(1)

file_path = sys.argv[1]

# Leggi i dati dal CSV
data = pd.read_csv(file_path, sep=';')

# Estrai i dati
versions = data['Version']
elapsed_time = data['Elapsed Time (s)']
speedup = data['Speedup']

# Grafico a barre per il tempo di esecuzione
plt.figure(figsize=(8,5))
plt.bar(versions, elapsed_time, color=['blue', 'green'])
plt.ylabel('Elapsed Time (s)')
plt.title('Execution Time Comparison')
for i, v in enumerate(elapsed_time):
    plt.text(i, v + 0.5, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.show()

# Grafico a barre per lo speedup
plt.figure(figsize=(8,5))
plt.bar(versions, speedup, color=['blue', 'green'])
plt.ylabel('Speedup')
plt.title('Speedup Comparison')
for i, v in enumerate(speedup):
    plt.text(i, v + 0.05, f"{v:.2f}", ha='center')
plt.tight_layout()
plt.show()
