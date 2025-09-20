import os
import re

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

image_root = "new_image_goodstyle"
images_per_row = 8

latex_lines = []

latex_lines.append(r"\documentclass[a4paper,12pt]{article}")
latex_lines.append(r"\usepackage{graphicx}")
latex_lines.append(r"\usepackage{subcaption}")
latex_lines.append(r"\usepackage[margin=1cm]{geometry}")
latex_lines.append(r"\usepackage{float}")
latex_lines.append(r"\begin{document}")
latex_lines.append(r"\section*{All Cluster Images}")

all_images = sorted(os.listdir(image_root), key=natural_sort_key)
for i, img in enumerate(all_images):
    if not img.endswith(".png") or "with_colorbar" not in img:
        continue
    img_path = os.path.join(image_root, img)
    if not os.path.exists(img_path):
        print("not found:", img_path)
        continue
    img_path = img_path.replace("\\", "/")

    if i % images_per_row == 0:
        latex_lines.append(r"\begin{figure}[H]")

    latex_lines.append(r"  \begin{subfigure}{0.11\textwidth}")
    latex_lines.append(f"    \\includegraphics[width=\\linewidth]{{{img_path}}}")
    latex_lines.append(r"  \end{subfigure}")
    latex_lines.append(r"  \hfill")

    if (i+1) % images_per_row == 0 or i == len(all_images)-1:
        latex_lines.append(r"\end{figure}")

latex_lines.append(r"\end{document}")

tex_file = "all_images.tex"
with open(tex_file, "w") as f:
    f.write("\n".join(latex_lines))

print(f"LaTeX: {tex_file}")
