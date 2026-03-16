from dataclasses import dataclass
from datasets import Dataset, DatasetDict
import numpy as np
from typing import Optional, Dict, Any
import json

def precompute_spatial_indices(
    dataset: Dataset,
    coordinate_key: str = "flatmap_coordinates",
    save_path: Optional[str] = None
) -> Dataset:
    """
    Precompute spatial sorting indices and add them to the dataset.
    
    Args:
        dataset: HuggingFace dataset
        coordinate_key: Key for spatial coordinates in dataset
        save_path: Optional path to save the preprocessed dataset
        
    Returns:
        Dataset with added spatial index columns
    """
    # Extract coordinates
    coordinates = np.array([sample[coordinate_key] for sample in dataset])
    
    # Compute sorted indices
    x_sorted_indices = np.argsort(coordinates[:, 0])
    y_sorted_indices = np.argsort(coordinates[:, 1])
    
    # Get sorted coordinates
    x_sorted = coordinates[x_sorted_indices, 0]
    y_sorted = coordinates[y_sorted_indices, 1]
    
    # Compute ranges
    spatial_metadata = {
        "x_min": float(x_sorted[0]),
        "x_max": float(x_sorted[-1]),
        "y_min": float(y_sorted[0]),
        "y_max": float(y_sorted[-1]),
        "coordinate_key": coordinate_key
    }
    
    # Add new columns to dataset
    dataset = dataset.add_column("x_sorted_indices", x_sorted_indices.tolist())
    dataset = dataset.add_column("y_sorted_indices", y_sorted_indices.tolist())
    
    # Store metadata in dataset info
    dataset.info.metadata = dataset.info.metadata or {}
    dataset.info.metadata["spatial_metadata"] = spatial_metadata
    
    if save_path:
        dataset.save_to_disk(save_path)
        
        # Save metadata separately for easy access
        metadata_path = f"{save_path}/spatial_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(spatial_metadata, f)
    
    return dataset

def preprocess_dataset_dict(
    dataset_dict: DatasetDict,
    coordinate_key: str = "flatmap_coordinates",
    save_path: Optional[str] = None
) -> DatasetDict:
    """Preprocess all splits in a dataset dictionary."""
    processed_datasets = {}
    
    for split, dataset in dataset_dict.items():
        split_save_path = f"{save_path}/{split}" if save_path else None
        processed_datasets[split] = precompute_spatial_indices(
            dataset, 
            coordinate_key=coordinate_key,
            save_path=split_save_path
        )
    
    if save_path:
        # Save the full dataset dictionary
        DatasetDict(processed_datasets).save_to_disk(save_path)
    
    return DatasetDict(processed_datasets)

# Modified PrecomputedData to use saved indices
@dataclass
class PrecomputedData:
    coordinates: np.ndarray
    x_sorted_indices: np.ndarray
    y_sorted_indices: np.ndarray
    x_range: float
    y_range: float
    window_width: float
    window_height: float

# Modify the sampler's _precompute_spatial_data method
def _precompute_spatial_data(self) -> PrecomputedData:
    """Use precomputed indices from dataset."""
    coordinates = np.array([
        self.dataset[i][self.coordinate_key] for i in range(len(self.dataset))
    ])
    
    # Get precomputed indices
    x_sorted_indices = np.array(self.dataset["x_sorted_indices"])
    y_sorted_indices = np.array(self.dataset["y_sorted_indices"])
    
    # Get metadata
    metadata = self.dataset.info.metadata["spatial_metadata"]
    x_range = metadata["x_max"] - metadata["x_min"]
    y_range = metadata["y_max"] - metadata["y_min"]
    
    return PrecomputedData(
        coordinates=coordinates,
        x_sorted_indices=x_sorted_indices,
        y_sorted_indices=y_sorted_indices,
        x_range=x_range,
        y_range=y_range,
        window_width=self.window_size * x_range,
        window_height=self.window_size * y_range
    )