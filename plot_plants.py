import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
input_configs = {
    'data/plant_measurements.csv': [
        'crown no.', 'leaf no.', 'inflorescence no.', 'flower no.',
        'fruit no.', 'petiole length mean (cm)', 'plant_FW (g)', 'plant_DW (g)',
        'shoots_DW (g)', 'leaf_DW (g)', 'leaf_area (cm2)', 'DW area-1',
    ],
    'data/fruit_quality.csv': ['acidity (meq 100gr -1)', 'brix (Bx%)', 'Bx acidity-1'],
    'data/fruit_measurements.csv': ['fruit_weight(g)', 'oidio (g)'],
    'data/firmness.csv': ['firmness (kg)'],
    'data/plant_yield.csv': ['yield_plant (g)']
}

# Optional: variable label mapping for prettier y-axis labels
variable_labels = {
    'inflorescence no.': 'Inflorescence Number',
    'flower no.': 'Flower Number',
    'fruit no.': 'Fruit Number',
    'leaf no.': 'Leaf Number',
    'petiole length mean (cm)': 'Petiole Length (cm)',
    'fruit_weight(g)': 'Fruit Weight (g)',
    'yield_plant (g)': 'Yield per Plant (g)',
    'acidity (meq 100gr -1)': 'Acidity (meq/100g)',
    'brix (Bx%)': 'Brix (%)',
    'Bx acidity-1': 'Brix/Acidity Ratio',
    'plant_FW (g)': 'Plant Fresh Weight (g)',
    'plant_DW (g)': 'Plant Dry Weight (g)',
    'shoots_DW (g)': 'Shoot Dry Weight (g)',
    'leaf_DW (g)': 'Leaf Dry Weight (g)',
    'leaf_area (cm2)': 'Leaf Area (cm²)',
    'DW area-1': 'Dry Weight per Area',
    'oidio (g)': 'Oidio (g)',
    'firmness (kg)': 'Firmness (kg)',
    'crown no.': 'Crown Number'
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
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df = df.dropna(subset=['date'])  # drop rows with invalid dates
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df.sort_values('date')

    # Prepare output folder
    file_name = os.path.splitext(os.path.basename(file_path))[0]
    plot_output_dir = os.path.join("line_plots", file_name)
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
            filtered = sub_df[['date_str', variable]].dropna()

            # Group by date and calculate mean ± std
            grouped = filtered.groupby('date_str').agg(
                mean=(variable, 'mean'),
                std=(variable, 'std')
            ).reset_index()

            if grouped.empty:
                continue

            face_color = 'black' if treatment == 'control' else 'white'
            edge_color = 'black'

            plt.errorbar(
                grouped['date_str'], grouped['mean'], yerr=grouped['std'],
                fmt=marker_styles.get(treatment, 'o'), linestyle='-',
                color=line_color,
                markerfacecolor=face_color,
                markeredgecolor=edge_color,
                label=treatment.capitalize(), capsize=4,
                elinewidth=1, linewidth=1.5, markersize=6
            )

        plt.xlabel('Date')
        plt.ylabel(label)
        plt.title(label)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        filename = f"{variable.replace(' ', '_').replace('/', '_')}_lineplot.png"
        plt.savefig(os.path.join(plot_output_dir, filename))
        plt.close()
