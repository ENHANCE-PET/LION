import os
import json
import zipfile
import requests
import shutil
import hashlib
import stat
import tempfile
from pathlib import Path, PurePosixPath
from lionz import system
from lionz.constants import (
    KEY_FOLDER_NAME,
    KEY_URL,
    KEY_LIMIT_FOV,
    DEFAULT_SPACING,
    FILE_NAME_DATASET_JSON,
    FILE_NAME_PLANS_JSON,
    TUMOR_LABEL,
    TRAINING_DATASET_SIZE_FDG,
    TRAINING_DATASET_SIZE_PSMA,
)


KEY_IMAGING_TYPE = "imaging_type"
KEY_MODALITY = "modality"
KEY_REQUIRED_MODALITIES = "required_modalities"
KEY_REQUIRED_PREFIXES = "required_prefixes"
KEY_NR_TRAINING = "nr_training_data"

MODEL_METADATA = {
    "psma": {
        KEY_URL: "https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.0/clin_pt_psma_2046_25112025.zip",
        KEY_FOLDER_NAME: "Dataset711_PSMA",
        "sha256": "3082cc95290a69ddd5c05a3fabcd573f5b45fb9dbf6ec1ccbc467acee2c0e3ee",
        TUMOR_LABEL: 6,
        KEY_IMAGING_TYPE: "clinical",
        KEY_MODALITY: "PT",
        KEY_REQUIRED_MODALITIES: ["PT"],
        KEY_REQUIRED_PREFIXES: ["PT_"],
        KEY_NR_TRAINING: TRAINING_DATASET_SIZE_PSMA,
    },
    "fdg": {
        KEY_URL: "https://github.com/ENHANCE-PET/LION/releases/download/lionz-v.1.0.3/clin_pt_fdg_5209_27032026.zip",
        KEY_FOLDER_NAME: "Dataset789_Tumors",
        "sha256": "e912620a316a4b35bef92912a3524715777da72dfb1de66951e75f21f5d44496",
        TUMOR_LABEL: 11,
        KEY_IMAGING_TYPE: "clinical",
        KEY_MODALITY: "PT",
        KEY_REQUIRED_MODALITIES: ["PT"],
        KEY_REQUIRED_PREFIXES: ["PT_"],
        KEY_NR_TRAINING: TRAINING_DATASET_SIZE_FDG,
    }
}

AVAILABLE_MODELS = MODEL_METADATA.keys()


class Model:
    def __init__(self, model_identifier: str, output_manager: system.OutputManager, base_directory: str | None = None):
        self.model_identifier = model_identifier
        self.folder_name = MODEL_METADATA[self.model_identifier][KEY_FOLDER_NAME]
        self.url = MODEL_METADATA[self.model_identifier][KEY_URL]
        self.sha256 = MODEL_METADATA[self.model_identifier]["sha256"]
        self.tumor_label = MODEL_METADATA[self.model_identifier][TUMOR_LABEL]
        self.limit_fov = False
        self.base_directory = base_directory or system.MODELS_DIRECTORY_PATH
        self.directory = os.path.join(self.base_directory, self.folder_name)

        self.__download(output_manager)
        self.configuration_folders = self.__get_configuration_folders(output_manager)
        self.configuration_directory = os.path.join(self.directory, self.configuration_folders[0])
        self.trainer, self.planner, self.resolution_configuration = self.__get_model_configuration()

        self.dataset, self.plans = self.__get_model_data()
        self.voxel_spacing = tuple(self.plans.get('configurations').get(self.resolution_configuration).get('spacing', DEFAULT_SPACING))
        self.imaging_type, self.modality = self.__get_model_identifier_segments()
        self.multilabel_prefix = f"{self.imaging_type}_{self.modality}"

        self.organ_indices = self.__get_organ_indices()
        self.nr_training_data = self.__get_number_training_data()

    def get_expectation(self):
        if self.modality == 'FDG-PET-CT':
            expected_modalities = ['FDG-PET', 'CT']
        else:
            expected_modalities = [self.modality]
        expected_prefixes = [m.replace('-', '_') + "_" for m in expected_modalities]

        return expected_modalities, expected_prefixes

    def __get_configuration_folders(self, output_manager: system.OutputManager) -> list[str]:
        items = os.listdir(self.directory)
        folders = [item for item in items if not item.startswith(".") and item.count("__") == 2 and os.path.isdir(os.path.join(self.directory, item))]

        if len(folders) > 1:
            output_manager.info("More than one configuration folder found. Using the first one encountered.")

        if not folders:
            raise ValueError(f"No valid configuration folders found in {self.directory}")

        return folders

    def __get_model_configuration(self) -> tuple[str, str, str]:
        model_configuration_folder = os.path.basename(self.configuration_directory)
        trainer, planner, resolution_configuration = model_configuration_folder.split("__")
        return trainer, planner, resolution_configuration

    def __get_model_identifier_segments(self) -> tuple[str, str]:
        metadata = MODEL_METADATA.get(self.model_identifier, {})
        imaging_type = metadata.get(KEY_IMAGING_TYPE, "clin")
        modality = metadata.get(KEY_MODALITY, "PT")
        return imaging_type, modality.upper()

    def __get_model_data(self) -> tuple[dict, dict]:
        dataset_json_path = os.path.join(self.configuration_directory, FILE_NAME_DATASET_JSON)
        plans_json_path = os.path.join(self.configuration_directory, FILE_NAME_PLANS_JSON)
        try:
            with open(dataset_json_path) as f:
                dataset = json.load(f)

            with open(plans_json_path) as f:
                plans = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Failed to load model data from {dataset_json_path} or {plans_json_path}: {e}")

        return dataset, plans

    def __download(self, output_manager: system.OutputManager):
        """
        Download and extract the model if it does not exist locally
        or if the existing folder has a different URL recorded.
        """

        # Reuse only caches carrying the expected URL and verified digest.
        if os.path.exists(self.directory):
            # Attempt to load the previously saved URL from version file
            version_file_path = os.path.join(self.directory, "model_version.json")
            old_url = None
            old_digest = None

            if os.path.exists(version_file_path):
                try:
                    with open(version_file_path, 'r') as vf:
                        version_data = json.load(vf)
                        old_url = version_data.get("url")
                        old_digest = version_data.get("sha256")
                except Exception:
                    pass  # If JSON is corrupted, we'll just treat it as mismatch

            # Preserve an old cache until its replacement has fully validated.
            if old_url != self.url or old_digest != self.sha256:
                output_manager.warn(f"Model version mismatch for '{self.model_identifier}', downloading latest...")
            else:
                # If the URL matches, we skip re-downloading
                output_manager.log_update(
                    f"    - A local instance of {self.model_identifier} has been detected."
                )
                output_manager.ok(f"{self.model_identifier.upper()} · Ready")
                return

        # Stage a replacement without modifying the current model.
        if not os.path.exists(self.base_directory):
            os.makedirs(self.base_directory)

        if not self.url:
            raise ValueError(f" No URL specified for model '{self.model_identifier}'.")

        output_manager.log_update(f"    - Downloading {self.model_identifier}")
        # A private temporary directory prevents planted download/extraction
        # symlinks in a shared model cache from redirecting writes.
        with tempfile.TemporaryDirectory(prefix=".lionz-model-", dir=self.base_directory) as staging:
            download_file_path = Path(staging) / "model.zip"
            digest = hashlib.sha256()
            with requests.get(self.url, stream=True, timeout=(10, 300)) as response:
                response.raise_for_status()
                total_size = int(response.headers.get("Content-Length", 0))
                progress = output_manager.create_file_progress_bar()
                with progress:
                    task = progress.add_task(f"[white] Downloading {self.model_identifier}...", total=total_size)
                    with download_file_path.open("xb") as output_file:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                digest.update(chunk)
                                output_file.write(chunk)
                                progress.update(task, advance=len(chunk))
            if digest.hexdigest() != self.sha256:
                raise ValueError(f"SHA-256 mismatch for model '{self.model_identifier}'")

            extraction_path = Path(staging) / "extracted"
            with zipfile.ZipFile(download_file_path) as archive:
                # Validate every extracted member before writing any content.
                # Apple-generated archives may contain a separate metadata tree;
                # ignore it entirely, including any unsafe names within it.
                members = []
                for member in archive.infolist():
                    path = PurePosixPath(member.filename)
                    if path.parts and (path.parts[0] == "__MACOSX" or member.filename == ".DS_Store"):
                        continue
                    mode = member.external_attr >> 16
                    if (path.is_absolute() or ".." in path.parts
                            or not path.parts or path.parts[0] != self.folder_name
                            or "\\" in member.filename or ":" in member.filename
                            or stat.S_ISLNK(mode)):
                        raise ValueError(f"Unsafe model archive member: {member.filename}")
                    members.append(member)
                archive.extractall(extraction_path, members=members)
            staged_model = extraction_path / self.folder_name
            if not staged_model.is_dir():
                raise ValueError("Model archive does not contain the expected directory")
            # Write the verification marker before exposing the new cache.
            with (staged_model / "model_version.json").open("w") as version_file:
                json.dump({"url": self.url, "sha256": self.sha256}, version_file)

            backup_root = Path(tempfile.mkdtemp(prefix=".lionz-backup-", dir=self.base_directory))
            backup_model = backup_root / "previous"
            try:
                if os.path.lexists(self.directory):
                    os.replace(self.directory, backup_model)
                try:
                    os.replace(staged_model, self.directory)
                except BaseException:
                    if os.path.lexists(backup_model):
                        # If rollback itself fails, leave this separate backup
                        # intact for recovery rather than deleting it on exit.
                        os.replace(backup_model, self.directory)
                    raise
            except BaseException:
                if not os.path.lexists(backup_model):
                    backup_root.rmdir()
                raise
            else:
                shutil.rmtree(backup_root)

        output_manager.log_update(f"    - {self.model_identifier} - setup complete.")
        output_manager.ok(f"{self.model_identifier.upper()} · Ready")
        
    def __get_organ_indices(self) -> dict[int, str]:
        labels = self.dataset.get('labels', {})
        return {int(value): key for key, value in labels.items() if value != "0"}

    def __get_number_training_data(self) -> str:
        nr_training_data = str(self.dataset.get('numTraining', "Not Available"))
        return nr_training_data

    def __str__(self):
        return self.model_identifier

    def __repr__(self):
        result = [
            f"Model Object of {self.model_identifier}",
            f" Folder Name: {self.folder_name}",
            f" URL: {self.url}",
            f" Directory: {self.directory}",
            f" Configuration Directory: {self.configuration_directory}",
            f" Trainer: {self.trainer}",
            f" Planner: {self.planner}",
            f" Resolution Configuration: {self.resolution_configuration}",
            f" Voxel Spacing: {self.voxel_spacing}",
            f" Imaging Type: {self.imaging_type}",
            f" Modality: {self.modality}",
            f" Region: {self.region}",
            f" Multilabel Prefix: {self.multilabel_prefix}",
            f" Organ Indices:",
        ]
        for index, organ in self.organ_indices.items():
            result.append(f"   {index}: {organ}")

        if isinstance(self.limit_fov, dict):
            result.append(f" Limit FOV:")
            for key, value in self.limit_fov.items():
                result.append(f"   {key}: {value}")
        else:
            result.append(f" Limit FOV: {self.limit_fov}")

        return "\n".join(result)

    @staticmethod
    def model_identifier_valid(model_identifier: str, output_manager: system.OutputManager) -> bool:
        if model_identifier not in MODEL_METADATA:
            output_manager.err("No valid model selected")
            return False

        model_information = MODEL_METADATA[model_identifier]
        if KEY_URL not in model_information or KEY_FOLDER_NAME not in model_information or KEY_LIMIT_FOV not in model_information:
            output_manager.err("Required keys url, folder_name, limit_fov missing")
            return False

        if model_information[KEY_URL] == "" or model_information[KEY_FOLDER_NAME] == "" or (model_information[KEY_LIMIT_FOV] is not None and not isinstance(model_information[KEY_LIMIT_FOV], dict)):
            output_manager.err("Required keys url, folder_name, limit_fov not defined correctly")
            return False

        return True


class ModelWorkflow:
    def __init__(self, model_identifier: str, output_manager: system.OutputManager):
        self.workflow: list[Model] = []
        self.__construct_workflow(model_identifier, output_manager)
        if self.workflow:
            self.initial_desired_spacing = self.workflow[0].voxel_spacing
            self.target_model = self.workflow[-1]

    def __construct_workflow(self, model_identifier: str, output_manager: system.OutputManager):
        model = Model(model_identifier, output_manager)
        if model.limit_fov and isinstance(model.limit_fov, dict) and 'model_to_crop_from' in model.limit_fov:
            self.__construct_workflow(model.limit_fov["model_to_crop_from"], output_manager)
        self.workflow.append(model)

    def __len__(self) -> len:
        return len(self.workflow)

    def __getitem__(self, index) -> Model:
        return self.workflow[index]

    def __iter__(self):
        return iter(self.workflow)

    def __str__(self) -> str:
        return " -> ".join([model.model_identifier for model in self.workflow])


def construct_model_routine(model_identifiers: str | list[str], output_manager: system.OutputManager) -> dict[tuple, list[ModelWorkflow]]:
    if isinstance(model_identifiers, str):
        model_identifiers = [model_identifiers]

    model_routine: dict = {}
    output_manager.log_update(' SETTING UP MODEL WORKFLOWS:')
    for model_identifier in model_identifiers:
        output_manager.log_update(' - Model name: ' + model_identifier)
        model_workflow = ModelWorkflow(model_identifier, output_manager)

        if model_workflow.initial_desired_spacing in model_routine:
            model_routine[model_workflow.initial_desired_spacing].append(model_workflow)
        else:
            model_routine[model_workflow.initial_desired_spacing] = [model_workflow]

    return model_routine
