import csv
import os

def generate_svg(data, filename="perf_visualization.svg"):
    width, height = 800, 600
    margin = 100
    bar_width = 150
    spacing = 80
    
    algorithms = [row[0] for row in data]
    times = [float(row[1]) for row in data]
    speedups = [float(row[2]) for row in data]
    
    max_time = max(times)
    
    chart_height = height - 2 * margin
    chart_width = width - 2 * margin
    
    def scale_y(val):
        return margin + chart_height - (val / max_time * chart_height)

    with open(filename, "w") as f:
        f.write(f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">\n')
        f.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
        
        # Title
        f.write(f'<text x="{width//2}" y="50" text-anchor="middle" font-family="sans-serif" font-size="24" font-weight="bold">SIMD Performance Comparison</text>\n')
        f.write(f'<text x="{width//2}" y="75" text-anchor="middle" font-family="sans-serif" font-size="14" fill="#666">Block Size: 512 samples | Average of 100,000 iterations</text>\n')
        
        colors = ["#3498db", "#e74c3c", "#2ecc71"]
        
        # Grid lines and Y-axis labels
        for i in range(5):
            y_val = max_time * i / 4
            y_pos = scale_y(y_val)
            f.write(f'<line x1="{margin}" y1="{y_pos}" x2="{width-margin}" y2="{y_pos}" stroke="#eee" />\n')
            f.write(f'<text x="{margin-10}" y="{y_pos+5}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#999">{y_val:.1f} us</text>\n')

        # Bars
        for i, (algo, time, speedup) in enumerate(zip(algorithms, times, speedups)):
            x = margin + i * (bar_width + spacing) + spacing//2
            y = scale_y(time)
            h = margin + chart_height - y
            
            # Bar with rounded corners
            f.write(f'<rect x="{x}" y="{y}" width="{bar_width}" height="{h}" fill="{colors[i % len(colors)]}" rx="5"/>\n')
            
            # Value on top of bar
            f.write(f'<text x="{x + bar_width//2}" y="{y - 10}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold" fill="{colors[i % len(colors)]}">{time:.3f} us</text>\n')
            
            # Algorithm name
            f.write(f'<text x="{x + bar_width//2}" y="{margin + chart_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="14" font-weight="bold">{algo}</text>\n')
            
            # Speedup text
            if speedup > 1.0:
                f.write(f'<text x="{x + bar_width//2}" y="{margin + chart_height + 45}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#666">{speedup:.1f}x faster</text>\n')
            else:
                f.write(f'<text x="{x + bar_width//2}" y="{margin + chart_height + 45}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#666">Baseline</text>\n')

        # X-axis line
        f.write(f'<line x1="{margin}" y1="{margin + chart_height}" x2="{width-margin}" y2="{margin + chart_height}" stroke="#ccc" stroke-width="2"/>\n')

        f.write('</svg>\n')

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(script_dir, "logs", "perf_results.csv")
    output_file = os.path.join(script_dir, "logs", "perf_visualization.svg")
    
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found. Run the C++ test first (from project root).")
    else:
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            data = list(reader)
        
        generate_svg(data, output_file)
        print(f"Successfully generated SVG visualization: {output_file} from {csv_file}")
