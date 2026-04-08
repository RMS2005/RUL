import json
import glob

output = []
for f in glob.glob('Mini Project 1/*.ipynb'):
    output.append(f"\n--- FILE: {f} ---")
    try:
        with open(f, encoding='utf-8', errors='ignore') as nb:
            data = json.load(nb)
            for cell in data.get('cells', []):
                if cell['cell_type'] == 'markdown':
                    text = "".join(cell.get('source', []))
                    if len(text.strip()) > 0:
                        output.append(text[:200].replace('\n', ' ')) # Extract first 200 chars of each markdown cell
    except Exception as e:
        output.append(f"Error: {e}")

with open('extracted_details.txt', 'w', encoding='utf-8') as f:
    f.write("\n".join(output))
