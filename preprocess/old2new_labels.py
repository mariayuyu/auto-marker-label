import btk
import os
import ast

def load_label_mapping(mapping_file):
    with open(mapping_file, 'r') as f:
        content = f.read()
        mapping = ast.literal_eval(content.split('=')[1].strip())
    return mapping

def rename_markers_in_c3d(input_path, output_path, label_mapping):
    reader = btk.btkAcquisitionFileReader()
    reader.SetFilename(input_path)
    reader.Update()

    acq = reader.GetOutput()
    points = acq.GetPoints()

    for i in range(points.GetItemNumber()):
        point = points.GetItem(i)
        old_label = point.GetLabel()
        if old_label in label_mapping:
            new_label = label_mapping[old_label]
            point.SetLabel(new_label)
            print(f"Renamed '{old_label}' to '{new_label}'")

    writer = btk.btkAcquisitionFileWriter()
    writer.SetInput(acq)
    writer.SetFilename(output_path)
    writer.Update()
    print(f"Saved updated C3D to {output_path}")


def process_c3d_folder(input_folder, output_folder, mapping_file):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    label_mapping = load_label_mapping(mapping_file)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(".c3d"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            rename_markers_in_c3d(input_path, output_path, label_mapping)

if __name__ == "__main__":
    input_folder = "input_c3d_files"
    output_folder = "output_c3d_files"
    mapping_file = "new_labels.txt"

    process_c3d_folder(input_folder, output_folder, mapping_file)