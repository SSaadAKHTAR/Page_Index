from mistralai.client import Mistral
import base64
import os
import glob
import re

MODEL = "pixtral-12b-2409"
API_KEY = "85rLAhDO5zwXHBFSkOWAa03OQcos5lPs"

def extract_image_definition(client, image_path, image_filename):
    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
        
    print(f"Extracting definition using API for: {image_filename}")
    response = client.chat.complete(
        model=MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            f"The image title is {image_filename}, Describe this image in detail. "
                            "If it is a technical diagram, explain all visible components, "
                            "connections, labels, arrows, and the purpose of the diagram."
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": f"data:image/png;base64,{image_b64}",
                    },
                ],
            }
        ],
    )
    return response.choices[0].message.content

def insert_definition_into_md(md_path, figure_label, definition):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    target_label = f"image: {figure_label}"
    # We want to insert the definition right after the line matching the figure label.
    inserted = False
    for i, line in enumerate(lines):
        if line.strip() == target_label:
            lines.insert(i + 1, f"\n{definition}\n\n")
            print(f"Inserted definition for '{figure_label}' at line {i + 1}.")
            inserted = True
            break
            
    if not inserted:
        print(f"Warning: Figure label '{figure_label}' not found alone on a single line in markdown.")

    with open(md_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)

def main():
    client = Mistral(api_key=API_KEY)
    image_dir = "/home/saad/Desktop/Page_Index/Image_defination/Images"
    md_path = "/home/saad/Desktop/Page_Index/Image_defination/UCIE_1.1.md"
    
    png_files = glob.glob(os.path.join(image_dir, "*.png"))
    # Sort files naturally if possible, we'll do basic sorting
    png_files.sort()
    
    count = 0
    for img_path in png_files:
        if count >= 10:
            print("Reached 10 images limit. Stopping.")
            break
            
        filename = os.path.basename(img_path)
        # Looking for something like "Figure 1-1" with an optional trailing dot.
        match = re.match(r"(Figure \d+-\d+)\.?", filename)
        if match:
            figure_label = match.group(1)
            definition = extract_image_definition(client, img_path, filename)
            insert_definition_into_md(md_path, figure_label, definition)
            count += 1
        else:
            print(f"Skipping {filename}: Could not extract a matching 'Figure X-X.' label.")

if __name__ == "__main__":
    main()