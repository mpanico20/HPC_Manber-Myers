import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv) < 5:
    print("Uso: python3 file.py <percorso_file_csv> <nome_serial> <nome_parallel> <string>")
    print("Esempio: python3 file.py dataset.csv O0 OpenMP_v1")
    sys.exit(1)

csv_file = sys.argv[1]
serial_name = sys.argv[2]
parallel_name = sys.argv[3] 
string_name = sys.argv[4]

if not os.path.exists(csv_file):
    raise FileNotFoundError(f"Il file '{csv_file}' non esiste")

df = pd.read_csv(csv_file, sep=';') 
df.columns = [col.strip().replace(' ', '_') for col in df.columns]

serial_time = df[df['Version'] == 'Serial']['Elapsed_Time_(s)'].iloc[0]
df['Speedup_vs_Serial'] = serial_time / df['Elapsed_Time_(s)']

def plot_speedup(df, serial_name, parallel_name):
    df_parallel = df[df['Version'] == parallel_name]
    num_threads = df_parallel['Num_of_thread'].values
    speedup_parallel = df_parallel['Speedup_vs_Serial'].values

    speedup_serial = [1] * len(num_threads)

    ideal = range(1, max(num_threads)+1)

    plt.figure(figsize=(10, 6))
    plt.plot(num_threads, speedup_parallel, marker='o', color='blue', label=parallel_name)
    plt.plot(num_threads, speedup_serial, marker='s', linestyle='', color='red', label=serial_name)
    plt.plot(range(1, max(num_threads)+1), ideal, linestyle='--', color='green', label='Ideal')

    plt.title(f'Speedup: {serial_name} vs {parallel_name}')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup vs Serial')
    plt.xticks(range(1, max(num_threads)+1))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    output_file = f'Graphs/speedup_vs_serial_{serial_name}_{parallel_name}_{string_name}.png'
    plt.savefig(output_file)

plot_speedup(df, serial_name, parallel_name)