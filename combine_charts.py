from PIL import Image

# Paths to your four images
paths = [
    "line_plots/leaf_no._lineplot.png",
    "line_plots/petiole_length_mean_(cm)_lineplot.png",
    "line_plots/inflorescence_no._lineplot.png",
    "line_plots/fruit_no._lineplot.png"
]

# Open all images
images = [Image.open(p) for p in paths]

# Optionally resize to same size (if not already the same)
widths, heights = zip(*(img.size for img in images))
max_width = max(widths)
max_height = max(heights)

# Resize images to the same size
images = [img.resize((max_width, max_height)) for img in images]

# Create a new blank canvas for 2x2 grid
grid_width = max_width * 2
grid_height = max_height * 2
combined = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

# Paste images into the grid
combined.paste(images[0], (0, 0))
combined.paste(images[1], (max_width, 0))
combined.paste(images[2], (0, max_height))
combined.paste(images[3], (max_width, max_height))

# Save result
combined.save("line_plots/combined_2x2.png")


# Paths to your four images
paths = [
    "line_plots/yield_plant_(g)_lineplot.png",
    "line_plots/fruit_weight(g)_lineplot.png",
    "line_plots/cumulative_plant_yield_(g)_lineplot.png",
    "line_plots/fruit_no._plant_lineplot.png"
]

# Open all images
images = [Image.open(p) for p in paths]

# Optionally resize to same size (if not already the same)
widths, heights = zip(*(img.size for img in images))
max_width = max(widths)
max_height = max(heights)

# Resize images to the same size
images = [img.resize((max_width, max_height)) for img in images]

# Create a new blank canvas for 2x2 grid
grid_width = max_width * 2
grid_height = max_height * 2
combined = Image.new("RGB", (grid_width, grid_height), (255, 255, 255))

# Paste images into the grid
combined.paste(images[0], (0, 0))
combined.paste(images[1], (max_width, 0))
combined.paste(images[2], (0, max_height))
combined.paste(images[3], (max_width, max_height))

# Save result
combined.save("line_plots/combined_2x2_2.png")
