"""
Dataset-specific configuration management for Hybrid Forensics
Handles thresholds and parameters for different datasets
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List


class DatasetConfig:
    """Manage dataset-specific configurations and thresholds"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize dataset configuration
        
        Parameters
        ----------
        config_path : str, optional
            Path to dataset thresholds YAML file
        """
        self.datasets = {}
        self.guidelines = {}
        self.metrics = {}
        
        if config_path is None:
            # Use default path
            config_path = self._get_default_config_path()
        
        if os.path.exists(config_path):
            self._load_config(config_path)
    
    def _get_default_config_path(self) -> str:
        """Get default dataset config path"""
        project_root = Path(__file__).parent.parent
        return str(project_root / 'configs' / 'dataset_thresholds.yaml')
    
    def _load_config(self, config_path: str) -> None:
        """Load dataset configuration from YAML file"""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if config:
            self.datasets = config.get('datasets', {})
            self.guidelines = config.get('tuning_guidelines', {})
            self.metrics = config.get('performance_metrics', {})
    
    def get_dataset_names(self) -> List[str]:
        """Get list of available datasets"""
        return list(self.datasets.keys())
    
    def get_thresholds(self, dataset_name: str) -> Dict[str, float]:
        """
        Get thresholds for a specific dataset
        
        Parameters
        ----------
        dataset_name : str
            Name of dataset (e.g., 'lrs2', 'lrw', 'fakeavceleb', 'avlips')
        
        Returns
        -------
        dict
            Dictionary with 'speech_threshold' and 'dynamic_threshold'
        
        Raises
        ------
        ValueError
            If dataset not found
        """
        if dataset_name not in self.datasets:
            available = ', '.join(self.get_dataset_names())
            raise ValueError(
                f"Dataset '{dataset_name}' not found. "
                f"Available datasets: {available}"
            )
        
        dataset_config = self.datasets[dataset_name]
        return {
            'speech_threshold': dataset_config.get('speech_threshold'),
            'dynamic_threshold': dataset_config.get('dynamic_threshold')
        }
    
    def get_speech_threshold(self, dataset_name: str) -> float:
        """Get SpeechForensics threshold for dataset"""
        return self.get_thresholds(dataset_name)['speech_threshold']
    
    def get_dynamic_threshold(self, dataset_name: str) -> float:
        """Get DynamicForensics threshold for dataset"""
        return self.get_thresholds(dataset_name)['dynamic_threshold']
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, Any]:
        """Get complete information about a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Dataset '{dataset_name}' not found")
        
        return self.datasets[dataset_name]
    
    def get_metrics(self, dataset_name: str) -> Dict[str, float]:
        """Get performance metrics for a dataset"""
        if dataset_name not in self.metrics:
            return {}
        
        return self.metrics[dataset_name]
    
    def print_all_datasets(self) -> None:
        """Print information about all available datasets"""
        print("\n" + "="*70)
        print("AVAILABLE DATASETS AND THRESHOLDS")
        print("="*70)
        
        for dataset_name, config in self.datasets.items():
            print(f"\n{dataset_name.upper()}")
            print("-" * 70)
            print(f"  Description: {config.get('description', 'N/A')}")
            print(f"  Speech Threshold: {config.get('speech_threshold')}")
            print(f"  Dynamic Threshold: {config.get('dynamic_threshold')}")
            print(f"  Notes: {config.get('notes', 'N/A')}")
            print(f"  Num Videos: {config.get('num_videos', 'N/A')}")
            
            # Print metrics if available
            if dataset_name in self.metrics:
                metrics = self.metrics[dataset_name]
                print(f"  Performance Metrics:")
                print(f"    AUC: {metrics.get('auc', 'N/A')}")
                print(f"    Accuracy: {metrics.get('accuracy', 'N/A')}")
                print(f"    Precision: {metrics.get('precision', 'N/A')}")
                print(f"    Recall: {metrics.get('recall', 'N/A')}")
                print(f"    F1: {metrics.get('f1', 'N/A')}")
        
        print("\n" + "="*70)
    
    def print_guidelines(self) -> None:
        """Print threshold tuning guidelines"""
        print("\n" + "="*70)
        print("THRESHOLD TUNING GUIDELINES")
        print("="*70)
        
        for threshold_name, guideline in self.guidelines.items():
            print(f"\n{threshold_name.upper()}")
            print("-" * 70)
            print(f"  Description: {guideline.get('description', 'N/A')}")
            print(f"  Range: {guideline.get('range', 'N/A')}")
            print(f"  Interpretation: {guideline.get('interpretation', 'N/A')}")
            print(f"  Lower Value: {guideline.get('lower_value', 'N/A')}")
            print(f"  Higher Value: {guideline.get('higher_value', 'N/A')}")
        
        print("\n" + "="*70)


# Global dataset config instance
_global_dataset_config = None


def get_dataset_config(config_path: Optional[str] = None) -> DatasetConfig:
    """Get or create global dataset configuration instance"""
    global _global_dataset_config
    
    if _global_dataset_config is None:
        _global_dataset_config = DatasetConfig(config_path)
    
    return _global_dataset_config


def get_dataset_thresholds(dataset_name: str) -> Dict[str, float]:
    """
    Convenience function to get thresholds for a dataset
    
    Parameters
    ----------
    dataset_name : str
        Dataset name (e.g., 'lrs2', 'lrw', 'fakeavceleb', 'avlips')
    
    Returns
    -------
    dict
        Dictionary with 'speech_threshold' and 'dynamic_threshold'
    """
    config = get_dataset_config()
    return config.get_thresholds(dataset_name)


if __name__ == '__main__':
    # Example usage
    config = DatasetConfig()
    
    print("Available datasets:", config.get_dataset_names())
    
    # Print all datasets
    config.print_all_datasets()
    
    # Print guidelines
    config.print_guidelines()
    
    # Get specific thresholds
    print("\nLRS2 Thresholds:", config.get_thresholds('lrs2'))
    print("LRW Thresholds:", config.get_thresholds('lrw'))
    print("FakeAVCeleb Thresholds:", config.get_thresholds('fakeavceleb'))
    print("AVLips Thresholds:", config.get_thresholds('avlips'))
