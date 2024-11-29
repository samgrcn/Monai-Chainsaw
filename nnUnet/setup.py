import os
import shutil
import gzip
import nibabel as nib
import json

# Setting up environment variables TO BE UPDATED
os.environ['nnUNet_raw'] = '/home/garcin/PycharmProjects/Monai-ESS/nnUnet/nnUNet_raw'
os.environ['nnUNet_preprocessed'] = '/home/garcin/PycharmProjects/Monai-ESS/nnUnet/nnUNet_preprocessed'
os.environ['nnUNet_results'] = '/home/garcin/PycharmProjects/Monai-ESS/nnUnet/nnUNet_results'

# Paths
original_data_dir = '../data/'
nnunet_data_dir = os.path.join(os.environ['nnUNet_raw'], 'Dataset001_FR')

imagesTr_dir = os.path.join(nnunet_data_dir, 'imagesTr')
labelsTr_dir = os.path.join(nnunet_data_dir, 'labelsTr')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)

# Loop through patient folders
for patient_id in range(1, 56):  # Assuming patient folders are named 1 to 55
    patient_folder = os.path.join(original_data_dir, str(patient_id))
    if not os.path.isdir(patient_folder):
        continue

    # Find the image file
    image_file = None
    for filename in os.listdir(patient_folder):
        if filename.startswith(' mDIXON-Quant_BH') and filename.endswith('.nii'):
            image_file = os.path.join(patient_folder, filename)
            break
        elif filename.startswith(' mDIXON-Quant_BH_v3') and filename.endswith('.nii'):
            image_file = os.path.join(patient_folder, filename)
            break
    if image_file is None:
        continue  # No image found

    label_file = os.path.join(patient_folder, 'erector.nii')
    if not os.path.exists(label_file):
        continue  # No label found

    # Define new filenames
    case_id = f'Case_{patient_id:03d}'
    image_dest = os.path.join(imagesTr_dir, f'{case_id}_0000.nii.gz')
    label_dest = os.path.join(labelsTr_dir, f'{case_id}.nii.gz')

    # Copy and compress the image
    img = nib.load(image_file)
    nib.save(img, image_dest)

    # Copy and compress the label
    lbl = nib.load(label_file)
    nib.save(lbl, label_dest)

# Create the dataset.json file according to nnU-Net v2 format
dataset = {
    "name": "Erector Spinae Segmentation",
    "description": "Segmentation of Erector Spinae muscle",
    "reference": "",  # Optional
    "licence": "",    # Optional
    "release": "0.0", # Optional
    "tensorImageSize": "3D",
    "channel_names": {
        "0": "MRI"
    },
    "labels": {
        "background": 0,
        "erector_spinae": 1
    },
    "numTraining": len(os.listdir(labelsTr_dir)),
    "file_ending": ".nii.gz",
    # "overwrite_image_reader_writer": "NibabelIO"  # Optional, uncomment if needed
}

with open(os.path.join(nnunet_data_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset, f, indent=4)
