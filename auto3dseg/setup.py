import os
import json
from monai.apps.auto3dseg import AutoRunner

# Define the root directory of your dataset
dataset_dir = "/Users/samuel/Documents/EPFL/BA5/Monai-ESS/data"

# Initialize the training data list
training_data = []

# Get the list of patient folders
patient_ids = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]

for patient_id in patient_ids:
    patient_dir = os.path.join(dataset_dir, patient_id)

    # Find the image file (handles both filenames)
    image_files = [
        f for f in os.listdir(patient_dir)
        if f.startswith(' mDIXON-Quant_BH') and f.endswith('.nii')
    ]
    if not image_files:
        print(f"No image file found for patient {patient_id}")
        continue  # Skip if no image file found
    else:
        image_file = image_files[0]  # Take the first match
        # Corrected line: use patient_dir instead of patient_id
        image_path = os.path.relpath(os.path.join(patient_dir, image_file), start=dataset_dir)

    # Find the label file (handles possible spaces at the beginning)
    label_files = [
        f for f in os.listdir(patient_dir)
        if f.strip() == 'erector.nii'
    ]
    if not label_files:
        print(f"No label file found for patient {patient_id}")
        continue  # Skip if no label file found
    else:
        label_file = label_files[0]  # Take the first match
        # Corrected line: use patient_dir instead of patient_id
        label_path = os.path.relpath(os.path.join(patient_dir, label_file), start=dataset_dir)

    # Append to training data
    training_data.append({
        "image": image_path,
        "label": label_path
    })

# Build the dataset dictionary
dataset = {
    "name": "Erector Segmentation",
    "description": "Dataset for erector muscle segmentation",
    "reference": "",
    "licence": "",
    "release": "0.0",
    "tensorImageSize": "3D",
    "modality": {
        "0": "MRI"
    },
    "labels": {
        "0": "background",
        "1": "erector"
    },
    "numTraining": len(training_data),
    "numTest": 0,
    "training": training_data,
    "test": []
}

# Save the dataset to a JSON file
data_list_file = os.path.join(dataset_dir, "dataset.json")
with open(data_list_file, 'w') as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset JSON saved to {data_list_file}")

# Set the work directory
work_dir = "/Users/samuel/Documents/EPFL/BA5/Monai-ESS/auto3dseg"

# Make sure the work_dir exists
os.makedirs(work_dir, exist_ok=True)

if __name__ == '__main__':
    # Create the AutoRunner
    runner = AutoRunner(
        work_dir=work_dir,
        input={
            "modality": "MRI",
            "datalist": data_list_file,
            "dataroot": dataset_dir
        }
    )

    # Run Auto3DSeg
    runner.run()