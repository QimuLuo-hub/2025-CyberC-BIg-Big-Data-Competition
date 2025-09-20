import os
import pandas as pd
from pathlib import Path

input_dir = Path("data/Au20_centered")

records = []

for fname in sorted(os.listdir(input_dir)):
    if fname.endswith(".xyz"):
        fpath = os.path.join(input_dir, fname)
        with open(fpath, "r") as f:
            lines = f.readlines()
            atom_num = int(lines[0].strip())
            total_energy = float(lines[1].strip())

        # extract the file id，e.g., "123_centered.xyz" -> "123"
        file_id = fname.split("_")[0].split(".")[0]

        records.append({"id": file_id, "atom number": atom_num, "total energy": total_energy})


df = pd.DataFrame(records).set_index("id")


output_path = Path("./atom_energy_summary.xlsx")
df.to_excel(output_path)

print("saved to", output_path)
