import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input_configs = {'data/plant_measurements.csv' :[
        'leaf no.', 'inflorescence no.', 'flower no.',
        'fruit no.', 'petiole length mean (cm)']}


# Optional: variable label mapping for prettier y-axis labels
variable_labels = {
    'inflorescence no.': 'Inflorescence Number',
    'flower no.': 'Flower Number',
    'fruit no.': 'Fruit Number',
    'leaf no.': 'Leaf Number',
    'petiole length mean (cm)': 'Petiole Length (cm)',
}

# Marker styles per treatment
marker_styles = {'control': 'o', 'shaded': 's'}
line_color = 'black'

# === PROCESS EACH FILE ===
for file_path, variables in input_configs.items():
    print(f"Processing: {file_path}")
    df = pd.read_csv(file_path)

    # Clean up columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Convert date
    df = df.sort_values('DAT')

    # Prepare output folder
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_output_dir = os.path.join("line_plots")
    os.makedirs(plot_output_dir, exist_ok=True)

    for variable in variables:
        if variable not in df.columns:
            print(f"  Skipping variable '{variable}' (not found in columns).")
            continue

        # Ensure numeric conversion
        df[variable] = pd.to_numeric(df[variable], errors='coerce')

        label = variable_labels.get(variable, variable)
        plt.figure(figsize=(8, 5))

        for treatment in df['treatment'].dropna().unique():
            sub_df = df[df['treatment'] == treatment]
            filtered = sub_df[['DAT', variable]].dropna()

            # Group by date and calculate mean ± std
            grouped = filtered.groupby('DAT').agg(
                mean=(variable, 'mean'),
                std=(variable, 'std')
            ).reset_index()

            grouped['DAT'] = grouped['DAT'].astype(str)

            if grouped.empty:
                continue

            face_color = 'black' if treatment == 'control' else 'white'
            edge_color = 'black'

            plt.errorbar(
            grouped['DAT'], grouped['mean'], yerr=grouped['std'],
            fmt=marker_styles.get(treatment, 'o'), linestyle='-',
            color=line_color,
            markerfacecolor=face_color,
            markeredgecolor=edge_color,
            label=treatment.capitalize(), capsize=4,
            elinewidth=1, linewidth=1.5, markersize=6
)

        plt.xlabel('DAT')
        plt.ylabel(label)
        plt.legend()
        plt.xticks(grouped['DAT'])
        plt.tight_layout()

        filename = f"{variable.replace(' ', '_').replace('/', '_')}_lineplot.png"
        plt.savefig(os.path.join(plot_output_dir, filename))
        plt.close()
