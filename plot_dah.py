import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input_configs = {
    'data/fruit_quality.csv': ['acidity (meq 100gr -1)', 'brix (Bx°)', 'Bx acidity-1'],
    'data/fruit_measurements.csv': ['fruit_weight(g)', 'oidio (g)'],
    'data/firmness.csv': ['firmness (kg)'],
    'data/plant_yield.csv': ['yield_plant (g)', 'fruit no. plant', 'cumulative_fruit no', 'cumulative_plant_yield (g)'],
}

# Optional: variable label mapping for prettier y-axis labels
variable_labels = {
    'fruit_weight(g)': 'average fruit weight (g)',
    'yield_plant (g)': 'fruit yield (g plant⁻¹)',
    'acidity (meq 100gr⁻¹)': 'Acidity (meq 100g⁻¹)',
    'brix (Bx%)': 'Brix (Bx°)',
    'Bx acidity-1': 'Brix/Acidity',
    'oidio (g)': 'Oidio (g)',
    'firmness (kg)': 'Firmness (kg)',
    'fruit no.': 'harvested fruit number',
    'fruit no. plant': 'harvested fruit number (plant⁻¹)',
    'cumulative_fruit no': 'cumulative fruit number',
    'cumulative_plant_yield (g)': 'cumulative fruit yield (g plant⁻¹)',
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
    df = df.sort_values('DAH')

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
            filtered = sub_df[['DAH', variable]].dropna()

            # Group by date and calculate mean ± std
            grouped = filtered.groupby('DAH').agg(
                mean=(variable, 'mean'),
                std=(variable, 'std')
            ).reset_index()

            grouped['DAH'] = grouped['DAH'].astype(str)

            if grouped.empty:
                continue

            face_color = 'black' if treatment == 'control' else 'white'
            edge_color = 'black'

            plt.errorbar(
            grouped['DAH'], grouped['mean'], yerr=grouped['std'],
            fmt=marker_styles.get(treatment, 'o'), linestyle='-',
            color=line_color,
            markerfacecolor=face_color,
            markeredgecolor=edge_color,
            label=treatment.capitalize(), capsize=4,
            elinewidth=1, linewidth=1.5, markersize=6
)

        plt.xlabel('DAH', fontsize=16)
        plt.ylabel(label, fontsize=20)
        plt.legend(fontsize=15)
        plt.xticks(grouped['DAH'], fontsize=16)
        plt.yticks(fontsize=16)

        filename = f"{variable.replace(' ', '_').replace('/', '_')}_lineplot.png"
        plt.savefig(os.path.join(plot_output_dir, filename), dpi=600)
        plt.close()
