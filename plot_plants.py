import os
import pandas as pd
import matplotlib.pyplot as plt

# === CONFIGURATION ===
data_files = [
    "data/plant_measurements.csv",
    "data/fruit_measurements.csv",
    "data/fruit_analysis.csv"
]

# Optional: Known variable labels (you can add more if needed)
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
    'Bx/acidity': 'Brix/Acidity Ratio',
}

# Marker styles per treatment
marker_styles = {'control': 'o', 'shaded': 's'}
line_color = 'black'

# === PROCESS EACH FILE ===
for file_path in data_files:
    df = pd.read_csv(file_path)
    file_name = os.path.splitext(os.path.basename(file_path))[0]

    # Prepare date formatting
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['date_str'] = df['date'].dt.strftime('%Y-%m-%d')
    df = df.sort_values('date')

    # Remove any unnamed columns (like index leftovers)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Create output folder
    plot_output_dir = os.path.join("statistics_output", "line_plots", file_name)
    os.makedirs(plot_output_dir, exist_ok=True)

    # Identify variables to plot (all numeric columns not metadata)
    excluded = ['date', 'date_str', 'treatment', 'tunnel', 'plant ID']
    numeric_vars = [col for col in df.columns if col not in excluded and pd.api.types.is_numeric_dtype(df[col])]

    # PLOT EACH VARIABLE
    for variable in numeric_vars:
        label = variable_labels.get(variable, variable)
        plt.figure(figsize=(6, 4))

        for treatment in df['treatment'].dropna().unique():
            sub_df = df[df['treatment'] == treatment]
            filtered = sub_df[['date_str', variable]].dropna()
            grouped = filtered.groupby('date_str').agg(
                mean=(variable, 'mean'),
                std=(variable, 'std')
            ).reset_index()

            if not grouped.empty:
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
