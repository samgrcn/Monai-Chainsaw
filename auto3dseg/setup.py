import os
import nibabel as nib
import json
from monai.apps.auto3dseg import BundleGen, DataAnalyzer, AutoRunner

# Paths
original_data_dir = '../data/'
auto3dseg_data_dir = '/Users/samuel/Documents/EPFL/BA5/Monai-ESS/auto3dseg/'
project_folder = os.path.join(auto3dseg_data_dir, 'auto3dseg_project')

imagesTr_dir = os.path.join(auto3dseg_data_dir, 'imagesTr')
labelsTr_dir = os.path.join(auto3dseg_data_dir, 'labelsTr')
os.makedirs(imagesTr_dir, exist_ok=True)
os.makedirs(labelsTr_dir, exist_ok=True)
os.makedirs(project_folder, exist_ok=True)

# Prepare dataset (copying images and labels)
for patient_id in range(1, 56):
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
    image_dest = os.path.join(imagesTr_dir, f'{case_id}.nii.gz')
    label_dest = os.path.join(labelsTr_dir, f'{case_id}.nii.gz')

    # Copy and compress the image
    img = nib.load(image_file)
    nib.save(img, image_dest)

    # Copy and compress the label
    lbl = nib.load(label_file)
    nib.save(lbl, label_dest)

# Create dataset.json
dataset = {
    "name": "Erector Spinae Segmentation",
    "description": "Segmentation of Erector Spinae muscle",
    "reference": "",
    "licence": "",
    "release": "0.0",
    "tensorImageSize": "3D",
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
            "label": f"./labelsTr/{f}"
        }
        for f in sorted(os.listdir(imagesTr_dir))
    ],
    "test": []
}

with open(os.path.join(auto3dseg_data_dir, 'dataset.json'), 'w') as f:
    json.dump(dataset, f, indent=4)

# Analyze the dataset
data_src_cfg = {
    "name": "ErectorSpinaeSegmentation",
    "task": "segmentation",
    "modality": "MRI",
    "datalist": os.path.join(auto3dseg_data_dir, 'dataset.json'),
    "dataroot": auto3dseg_data_dir,
}

analyzer = DataAnalyzer(data_src_cfg, output_path=project_folder)
analyzer.get_all_case_stats()

# Generate algorithm templates
bundle_generator = BundleGen(
    algo_path=project_folder,
    data_stats_filename=os.path.join(project_folder, "datastats.yaml"),
    templates_path_or_url=None,  # Use default templates
    data_src_cfg_name="data_src_cfg.yaml",
)

algos = bundle_generator.generate()

# Run training
runner = AutoRunner(
    algo_path=project_folder,
    datastats_filename=os.path.join(project_folder, "datastats.yaml"),
    data_src_cfg_name="data_src_cfg.yaml",
)

runner.run()

# Ensemble models (optional)
from monai.apps.auto3dseg import AlgoEnsembleBuilder

ensemble_builder = AlgoEnsembleBuilder(
    algo_path=project_folder,
    data_stats_filename=os.path.join(project_folder, "datastats.yaml"),
    ensemble_method_name="AlgoEnsembleBestN",
)

ensemble_builder.set_ensemble_method(ensemble_method_name="AlgoEnsembleBestN")
ensemble = ensemble_builder.get_ensemble()