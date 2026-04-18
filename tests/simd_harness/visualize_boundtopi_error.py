import csv
import os

def generate_svg(data, filename="boundtopi_error_visualization.svg"):
    width, height = 1000, 600
    margin = 60
    plot_height = height - 2 * margin
    plot_width = width - 2 * margin

    try:
        data = sorted(data, key=lambda row: float(row[0]))
        x_vals = [float(row[0]) for row in data]
        diff_vals = [float(row[3]) for row in data]
    except (ValueError, IndexError) as e:
        print(f"Error parsing CSV data: {e}")
        return

    min_x, max_x = min(x_vals), max(x_vals)
    max_diff = max(diff_vals)
    if max_diff == 0: max_diff = 1e-10
    
    def scale_x(val):
        return margin + (val - min_x) / (max_x - min_x) * plot_width
    def scale_y(val):
        return margin + plot_height - ((val / max_diff) * plot_height)

    with open(filename, "w") as f:
        f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        
        # Title
        f.write(f'<text x="{width//2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="bold">boundToPi Scalar vs SIMD Validation</text>\n')

        # Plot
        f.write(f'<text x="{margin}" y="{margin-15}" font-family="sans-serif" font-size="16" font-weight="bold">Absolute Difference (Max: {max_diff:.2e})</text>\n')
        f.write(f'<rect x="{margin}" y="{margin}" width="{plot_width}" height="{plot_height}" fill="#fdfdfd" stroke="#ccc"/>\n')
        
        # Grid
        for i in range(5):
            val = (4 - i) * max_diff / 4
            y = margin + i * plot_height / 4
            f.write(f'<line x1="{margin}" y1="{y}" x2="{margin+plot_width}" y2="{y}" stroke="#eee" />\n')
            f.write(f'<text x="{margin-5}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#666">{val:.1e}</text>\n')

        # Path
        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in zip(x_vals, diff_vals)]) + '" fill="none" stroke="#2ecc71" stroke-width="2" />\n')

        # X-axis labels
        for x_val in [min_x, (min_x + max_x)/2, max_x]:
            f.write(f'<text x="{scale_x(x_val)}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">{x_val:.2f}</text>\n')
        f.write(f'<text x="{width//2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold">Input angle (rad)</text>\n')

        f.write('</svg>\n')

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "logs", "simd_boundtopi_results.csv")
    output_file = os.path.join(script_dir, "logs", "boundtopi_error_visualization.svg")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the C++ test first (from project root).")
    else:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader) 
            data = list(reader)
        
        generate_svg(data, output_file)
        print(f"Successfully generated SVG visualization: {output_file} from {csv_file}")
