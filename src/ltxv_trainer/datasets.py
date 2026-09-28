# Adapted from https://github.com/a-r-r-o-w/finetrainers/blob/main/finetrainers/dataset.py

from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from ltxv_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(1, self.latent_sequence_length, self.latent_dim),  # random video latents
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),  # random text embeddings
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),  # random attention mask
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(
        self, 
        data_root: str, 
        data_sources: dict[str, str] | list[str] | None = None,
        manifest_path: str | Path | None = None
    ) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.

        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
                         - Dict mapping directory names to output keys
                         - List of directory names (keys will equal values)
                         - None (defaults to ["latents", "conditions"])
            manifest_path: Optional path to a specific manifest file. 
                          If provided, it overrides the default dataset.json search.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths = self._setup_source_paths()
        self.manifest_path = Path(manifest_path) if manifest_path else None
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root)

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed sub-directory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> dict[str, Path]:
        """Map data source names to their actual directory paths."""
        source_paths = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name
            source_paths[dir_name] = source_path

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

        return source_paths

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources."""
        # Check for manifest file
        # Priority 1: Explicitly provided manifest path
        if self.manifest_path:
            if not self.manifest_path.exists():
                raise FileNotFoundError(f"Dataset manifest file not found: {self.manifest_path}")
            return self._load_from_manifest(self.manifest_path)

        # Use first data source as the reference to discover samples
        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]
        data_files = list(data_path.glob("**/*.pt"))

        if not data_files:
            raise ValueError(f"No data files found in {data_path}")

        # Initialize sample files dict
        sample_files = {output_key: [] for output_key in self.data_sources.values()}

        # For each data file, find corresponding files in other sources
        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check if corresponding files exist in ALL sources
            if self._all_source_files_exist(data_file, rel_path):
                self._fill_sample_data_files(data_file, rel_path, sample_files)

        return sample_files

    def _load_from_manifest(self, manifest_path: Path) -> dict[str, list[Path]]:
        """Load samples from a manifest file."""
        import json
        logger.info(f"Loading dataset from manifest: {manifest_path}")
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
            
        sample_files = {output_key: [] for output_key in self.data_sources.values()}
        
                # Track media path counts for inferring condition filenames (handling many-to-one)
        # media_path_counts = {}
        
        for entry in manifest:
            # Check if all required sources are present in the entry
            valid_entry = True
            entry_paths = {}
            
            # Check if this is a source manifest (has media_path but not latents/conditions)
            # We only attempt inference if standard keys are missing but media_path is present
            is_source_manifest = "media_path" in entry and "latents" not in entry
            
            if is_source_manifest:
                media_path = entry["media_path"]
                # idx = media_path_counts.get(media_path, 0)
                # media_path_counts[media_path] = idx + 1
                
                # Infer paths
                # Latents: media_path with .pt suffix
                latent_rel_path = Path(media_path).with_suffix(".pt")
                
                # Conditions: Determine path
                # We assume exact mapping (no suffixes) as requested.
                # This implies that if multiple captions map to the same video, 
                # they will all map to the same condition file (e.g. video.pt).
                condition_rel_path = Path(media_path).with_suffix(".pt")
                
                # Verify existence and map to output keys
                # We assume standard keys "latents" and "conditions" for source manifest inference
                
                # Check latents
                if "latents" in self.data_sources:
                    full_path = self.source_paths["latents"] / latent_rel_path
                    if not full_path.exists():
                        logger.warning(f"Inferred latent file not found: {full_path}")
                        valid_entry = False
                    else:
                        entry_paths[self.data_sources["latents"]] = latent_rel_path
                
                # Check conditions
                if "conditions" in self.data_sources:
                    full_path = self.source_paths["conditions"] / condition_rel_path
                    if not full_path.exists():
                        logger.warning(f"Inferred condition file not found: {full_path}")
                        valid_entry = False
                    else:
                        entry_paths[self.data_sources["conditions"]] = condition_rel_path
                
                # Verify existence and map to output keys
                # We assume standard keys "latents" and "conditions" for source manifest inference
                
                # Check latents
                if "latents" in self.data_sources:
                    full_path = self.source_paths["latents"] / latent_rel_path
                    if not full_path.exists():
                        logger.warning(f"Inferred latent file not found: {full_path}")
                        valid_entry = False
                    else:
                        entry_paths[self.data_sources["latents"]] = latent_rel_path
                
                # Check conditions
                if "conditions" in self.data_sources:
                    full_path = self.source_paths["conditions"] / condition_rel_path
                    if not full_path.exists():
                        logger.warning(f"Inferred condition file not found: {full_path}")
                        valid_entry = False
                    else:
                        entry_paths[self.data_sources["conditions"]] = condition_rel_path
                        
                # If we have other data sources that we can't infer, fail
                if len(entry_paths) != len(self.data_sources):
                     # Only warn if we haven't already warned about a missing file
                     if valid_entry:
                        logger.warning(f"Could not infer all data sources for entry: {entry}")
                        valid_entry = False

            else:
                for dir_name, output_key in self.data_sources.items():
                    if dir_name not in entry:
                        # If not in manifest, try to infer from latents if possible (legacy support?)
                        # But manifest should be complete.
                        logger.warning(f"Manifest entry missing key '{dir_name}': {entry}")
                        valid_entry = False
                        break
                    
                    path = Path(entry[dir_name])
                    full_path = self.source_paths[dir_name] / path
                    if not full_path.exists():
                        logger.warning(f"File not found: {full_path}")
                        valid_entry = False
                        break
                    
                    entry_paths[output_key] = path
            
            if valid_entry:
                for output_key, path in entry_paths.items():
                    sample_files[output_key].append(path)
                    
        logger.info(f"Loaded {len(next(iter(sample_files.values())))} samples from manifest")
        return sample_files

    def _all_source_files_exist(self, data_file: Path, rel_path: Path) -> bool:
        """Check if corresponding files exist in all data sources."""
        for dir_name in self.data_sources:
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            if not expected_path.exists():
                logger.warning(
                    f"No matching {dir_name} file found for: {data_file.name} (expected in: {expected_path})"
                )
                return False

        return True

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _fill_sample_data_files(self, data_file: Path, rel_path: Path, sample_files: dict[str, list[Path]]) -> None:
        """Add a valid sample to the sample_files tracking."""
        for dir_name, output_key in self.data_sources.items():
            expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
            sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        if not self.sample_files:
            raise ValueError("No valid samples found - all data sources must have matching files")

        # Verify all output keys have the same number of samples
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            source_path = self.source_paths[dir_name]
            file_rel_path = self.sample_files[output_key][index]
            file_path = source_path / file_rel_path

            try:
                data = torch.load(file_path, map_location="cpu", weights_only=True)
                result[output_key] = data
            except Exception as e:
                raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result
