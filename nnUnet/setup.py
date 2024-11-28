import os
import shutil
import gzip
import nibabel as nib
import json

#Setting up environment variables TO BE UPDATED
os.environ['nnUNet_raw_data_base'] = '/Users/samuel/Documents/EPFL/BA5/Monai-ESS/nnUnet/nnUNet_raw_data_base'
os.environ['nnUNet_preprocessed'] = '/Users/samuel/Documents/EPFL/BA5/Monai-ESS/nnUnet/nnUNet_preprocessed'
os.environ['RESULTS_FOLDER'] = '/Users/samuel/Documents/EPFL/BA5/Monai-ESS/nnUnet/nnUNet_trained_models'

# Paths
original_data_dir = '../data/'
nnunet_data_dir = 'nnUNet_raw_data_base/nnUNet_raw_data/Task501_ErectorSpinaeSegmentation/'

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

dataset = {
    "name": "Erector Spinae Segmentation",
    "description": "Segmentation of Erector Spinae muscle",
    "tensorImageSize": "3D",
    "reference": "",
    "licence": "",
    "release": "0.0",
    "modality": {
        "0": "MRI"
    },
    "labels": {
        "0": "background",
        "1": "erector_spinae"
    },
    "numTraining": len(os.listdir(labelsTr_dir)),
    "numTest": 0,
    "training": [
        {
            "image": f"./imagesTr/{f}",
            "label": f"./labelsTr/{f.replace('_0000.nii.gz', '.nii.gz')}"
        }
        for f in sorted(os.listdir(imagesTr_dir))
    ],
    "test": []
}

with open(os.path.join(nnunet_data_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset, f, indent=4)



