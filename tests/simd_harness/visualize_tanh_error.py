import csv
import os

def generate_svg(data, filename="tanh_error_visualization.svg"):
    width, height = 1000, 800
    margin = 60
    panel_height = (height - 3 * margin) // 2
    plot_width = width - 2 * margin

    try:
        data = sorted(data, key=lambda row: float(row[0]))
        x_vals = [float(row[0]) for row in data]
        std_vals = [float(row[1]) for row in data]
        scalar_vals = [float(row[2]) for row in data]
        simd_vals = [float(row[3]) for row in data]
        simd_bounded_vals = [float(row[4]) for row in data]
        scalar_err = [float(row[5]) for row in data]
        simd_err = [float(row[6]) for row in data]
        simd_bounded_err = [float(row[7]) for row in data]
    except (ValueError, IndexError) as e:
        print(f"Error parsing CSV data: {e}")
        return

    min_x, max_x = min(x_vals), max(x_vals)
    
    # Top panel: tanh(x)
    min_y1, max_y1 = -1.2, 1.2
    def scale_x(val):
        return margin + (val - min_x) / (max_x - min_x) * plot_width
    def scale_y1(val):
        return margin + panel_height - ((val - min_y1) / (max_y1 - min_y1) * panel_height)

    # Bottom panel: error 
    max_err = max(max(scalar_err), max(simd_err), max(simd_bounded_err))
    if max_err == 0: max_err = 1e-10
    
    def scale_y2(val):
        start_y2 = margin * 2 + panel_height
        return start_y2 + panel_height - ((val / max_err) * panel_height)

    with open(filename, "w") as f:
        f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        
        # Title
        f.write(f'<text x="{width//2}" y="35" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="bold">SIMD Pade Tanh Approximant Validation</text>\n')

        # Top Panel
        f.write(f'<text x="{margin}" y="{margin-15}" font-family="sans-serif" font-size="16" font-weight="bold">tanh(x) Comparison</text>\n')
        f.write(f'<rect x="{margin}" y="{margin}" width="{plot_width}" height="{panel_height}" fill="#fdfdfd" stroke="#ccc"/>\n')
        
        # Grid for top panel
        for i in range(5):
            val = max_y1 - i * (max_y1 - min_y1) / 4
            y = margin + i * panel_height / 4
            f.write(f'<line x1="{margin}" y1="{y}" x2="{margin+plot_width}" y2="{y}" stroke="#eee" />\n')
            f.write(f'<text x="{margin-5}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#666">{val:g}</text>\n')

        # Paths
        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y1(y):.2f}" for x, y in zip(x_vals, std_vals)]) + '" fill="none" stroke="#3498db" stroke-width="3" opacity="0.4" />\n')
        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y1(y):.2f}" for x, y in zip(x_vals, scalar_vals)]) + '" fill="none" stroke="#e74c3c" stroke-width="1.5" stroke-dasharray="5,3" />\n')
        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y1(y):.2f}" for x, y in zip(x_vals, simd_vals)]) + '" fill="none" stroke="#2ecc71" stroke-width="1.5" stroke-dasharray="2,2" />\n')

        # Legend 1
        f.write(f'<rect x="{margin+10}" y="{margin+10}" width="160" height="65" fill="white" fill-opacity="0.8" stroke="#ccc"/>\n')
        f.write(f'<text x="{margin+20}" y="{margin+30}" fill="#3498db" font-family="sans-serif" font-size="12">Solid: std::tanh</text>\n')
        f.write(f'<text x="{margin+20}" y="{margin+45}" fill="#e74c3c" font-family="sans-serif" font-size="12">Dash: Pade Scalar</text>\n')
        f.write(f'<text x="{margin+20}" y="{margin+60}" fill="#2ecc71" font-family="sans-serif" font-size="12">Dot: Pade SIMD</text>\n')

        # Bottom Panel
        start_y2 = margin * 2 + panel_height
        f.write(f'<text x="{margin}" y="{start_y2-15}" font-family="sans-serif" font-size="16" font-weight="bold">Absolute Error (Max: {max_err:.2e})</text>\n')
        f.write(f'<rect x="{margin}" y="{start_y2}" width="{plot_width}" height="{panel_height}" fill="#fdfdfd" stroke="#ccc"/>\n')

        # Grid for bottom panel
        for i in range(5):
            val = (4 - i) * max_err / 4
            y = start_y2 + i * panel_height / 4
            f.write(f'<line x1="{margin}" y1="{y}" x2="{margin+plot_width}" y2="{y}" stroke="#eee" />\n')
            f.write(f'<text x="{margin-5}" y="{y+5}" text-anchor="end" font-family="sans-serif" font-size="10" fill="#666">{val:.1e}</text>\n')

        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y2(y):.2f}" for x, y in zip(x_vals, scalar_err)]) + '" fill="none" stroke="#e74c3c" stroke-width="1.5" />\n')
        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y2(y):.2f}" for x, y in zip(x_vals, simd_err)]) + '" fill="none" stroke="#2ecc71" stroke-width="1.5" stroke-dasharray="3,3" />\n')
        f.write('<path d="M' + ' L'.join([f"{scale_x(x):.2f},{scale_y2(y):.2f}" for x, y in zip(x_vals, simd_bounded_err)]) + '" fill="none" stroke="#f39c12" stroke-width="1" stroke-dasharray="1,1" />\n')

        # X-axis labels
        for x_val in [min_x, (min_x + max_x)/2, max_x]:
            f.write(f'<text x="{scale_x(x_val)}" y="{height-margin+20}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#666">{x_val:.2f}</text>\n')
        f.write(f'<text x="{width//2}" y="{height-10}" text-anchor="middle" font-family="sans-serif" font-size="12" font-weight="bold">Input x</text>\n')

        # Legend 2
        f.write(f'<rect x="{margin+10}" y="{start_y2+10}" width="160" height="80" fill="white" fill-opacity="0.8" stroke="#ccc"/>\n')
        f.write(f'<text x="{margin+20}" y="{start_y2+30}" fill="#e74c3c" font-family="sans-serif" font-size="12">Red: Scalar Error</text>\n')
        f.write(f'<text x="{margin+20}" y="{start_y2+45}" fill="#2ecc71" font-family="sans-serif" font-size="12">Green: SIMD Error</text>\n')
        f.write(f'<text x="{margin+20}" y="{start_y2+60}" fill="#f39c12" font-family="sans-serif" font-size="12">Orange: Bounded Error</text>\n')

        f.write('</svg>\n')

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "logs", "simd_tanh_results.csv")
    output_file = os.path.join(script_dir, "logs", "tanh_error_visualization.svg")

    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the C++ test first (from project root).")
    else:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader) 
            data = list(reader)
        
        generate_svg(data, output_file)
        print(f"Successfully generated SVG visualization: {output_file} from {csv_file}")
