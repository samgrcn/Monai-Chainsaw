import os
import json
import nibabel as nib
from sklearn.model_selection import train_test_split
from monai.apps.auto3dseg import AutoRunner
import logging

# Suppress INFO and DEBUG logs from MONAI to prevent TypeError in logging
logging.getLogger('monai').setLevel(logging.WARNING)


def prepare_dataset(original_data_dir, imagesTr_dir, labelsTr_dir):
    """
    Copies and renames image and label files from the original data directory
    to imagesTr and labelsTr directories with proper compression.
    """
    os.makedirs(imagesTr_dir, exist_ok=True)
    os.makedirs(labelsTr_dir, exist_ok=True)

    # Get sorted list of patient folders based on numerical order
    patient_ids = sorted(
        [folder for folder in os.listdir(original_data_dir) if os.path.isdir(os.path.join(original_data_dir, folder))],
        key=lambda x: int(x)
    )

    for idx, patient_id in enumerate(patient_ids, start=1):
        patient_folder = os.path.join(original_data_dir, patient_id)

        # Initialize variables
        image_file = None

        # Search for the image file with leading space
        for filename in os.listdir(patient_folder):
            if (filename.startswith(' mDIXON-Quant_BH') or filename.startswith(' mDIXON-Quant_BH_v3')) and (
                    filename.endswith('.nii') or filename.endswith('.nii.gz')):
                image_file = os.path.join(patient_folder, filename)
                break

        if not image_file:
            continue  # Skip if no image found

        # Define label file path
        label_file = os.path.join(patient_folder, 'erector.nii')
        if not os.path.exists(label_file):
            continue  # Skip if no label found

        # Define new standardized filenames
        case_id = f'Case_{idx:03d}.nii.gz'
        image_dest = os.path.join(imagesTr_dir, case_id)
        label_dest = os.path.join(labelsTr_dir, case_id)

        # Process and save image file
        try:
            img = nib.load(image_file)
            nib.save(img, image_dest)
        except Exception as e:
            print(f"Error processing image {image_file}: {e}")
            continue  # Skip if there's an issue with the image file

        # Process and save label file
        try:
            lbl = nib.load(label_file)
            nib.save(lbl, label_dest)
        except Exception as e:
            print(f"Error processing label {label_file}: {e}")
            continue  # Skip if there's an issue with the label file


def create_dataset_json(imagesTr_dir, labelsTr_dir, dataset_json_path, split_ratio=0.2):
    """
    Creates dataset.json with training and validation splits using relative paths.
    """
    # List all image and label files
    image_files = sorted([f for f in os.listdir(imagesTr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])
    label_files = sorted([f for f in os.listdir(labelsTr_dir) if f.endswith('.nii') or f.endswith('.nii.gz')])

    # Identify common files present in both directories
    common_files = list(set(image_files).intersection(set(label_files)))
    common_files.sort()

    if not common_files:
        raise ValueError("No matching image and label files found.")

    # Split into training and validation sets
    train_files, val_files = train_test_split(common_files, test_size=split_ratio, random_state=42)

    # Define dataset configuration
    dataset = {
        "name": "ErectorSpinaeSegmentation",
        "description": "Segmentation of Erector Spinae muscle",
        "reference": "",
        "licence": "",
        "release": "0.0",
        "tensorImageSize": "3D",
        "modality": {
            "0": "MRI"  # Change to "CT" if your data is CT
        },
        "labels": {
            "0": "background",
            "1": "erector_spinae"  # Adjust if you have different labels
        },
        "numTraining": len(train_files),
        "numTest": 0,
        "training": [
            {
                "image": f"./imagesTr/{f}",
                "label": f"./labelsTr/{f}"
            }
            for f in train_files
        ],
        "validation": [
            {
                "image": f"./imagesTr/{f}",
                "label": f"./labelsTr/{f}"
            }
            for f in val_files
        ],
        "test": []
    }

    # Save dataset.json
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset, f, indent=4)


def run_autorunner(work_dir, dataset_json_path):
    """
    Runs the Auto3Dseg pipeline using AutoRunner.
    """
    input_config = {
        "modality": "MRI",  # Change to "CT" if needed
        "datalist": dataset_json_path,
        "dataroot": work_dir,
    }

    runner = AutoRunner(
        work_dir=work_dir,
        input=input_config,
    )

    runner.run()


def main():
    """
    Main function to prepare the dataset and run Auto3Dseg.
    """
    # Define paths
    script_dir = os.path.dirname(os.path.abspath(__file__))  # Monai-ESS/auto3dseg/
    original_data_dir = '/Users/samuel/Documents/EPFL/BA5/Monai-ESS/data'  # Update this path if different
    imagesTr_dir = os.path.join(script_dir, 'imagesTr')
    labelsTr_dir = os.path.join(script_dir, 'labelsTr')
    dataset_json_path = os.path.join(script_dir, 'dataset.json')

    # Step 1: Prepare dataset (copy and rename)
    print("Preparing dataset...")
    prepare_dataset(original_data_dir, imagesTr_dir, labelsTr_dir)

    # Step 2: Create dataset.json
    print("Creating dataset.json...")
    create_dataset_json(imagesTr_dir, labelsTr_dir, dataset_json_path, split_ratio=0.2)

    # Step 3: Run Auto3Dseg
    print("Running Auto3Dseg pipeline...")
    run_autorunner(script_dir, dataset_json_path)

    print("Pipeline completed.")


if __name__ == "__main__":
    main()