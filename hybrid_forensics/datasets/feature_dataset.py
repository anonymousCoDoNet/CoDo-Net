import os
import os.path as osp
from typing import Dict, Tuple, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def _to_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().clone()
    else:
        tensor = torch.as_tensor(value)
    return tensor.float()


def _resample_sequence(sequence: torch.Tensor, target_len: Optional[int]) -> torch.Tensor:
    if target_len is None or sequence.size(0) == target_len:
        return sequence
    if sequence.size(0) == 1:
        return sequence.repeat(target_len, 1)
    seq = sequence.unsqueeze(0).transpose(1, 2)  # [1, C, T]
    seq = F.interpolate(seq, size=target_len, mode="linear", align_corners=False)
    seq = seq.transpose(1, 2).squeeze(0)
    return seq


class FeatureDataset(Dataset):
    """
    Dataset that loads cached Speech/Lip features produced by extract_features.

    Now supports:
    - Concat features (visual + audio = 2048-dim) for ImprovedCoDoNet
    - Padding masks for variable-length sequences
    - Optional data augmentation
    """

    def __init__(
        self,
        file_list: str,
        feature_root: str,
        seq_len: int = 64,
        require_dynamic: bool = True,
        include_audio: bool = False,
        use_concat: bool = True,
        create_mask: bool = True,
        augment: bool = False,
    ) -> None:
        super().__init__()
        self.feature_root = feature_root
        self.seq_len = seq_len
        self.require_dynamic = require_dynamic
        self.include_audio = include_audio
        self.use_concat = use_concat
        self.create_mask = create_mask
        self.augment = augment

        self.samples = []
        with open(file_list, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 2:
                    continue
                rel_path, label_str = parts[:2]
                rel_no_ext = osp.splitext(rel_path)[0]
                sample_dir = osp.join(feature_root, rel_no_ext)
                speech_path = osp.join(sample_dir, "speech_features.pt")
                dynamic_path = osp.join(sample_dir, "dynamic_features.pt")

                if not osp.exists(speech_path):
                    continue
                if require_dynamic and not osp.exists(dynamic_path):
                    continue

                self.samples.append(
                    {
                        "video": rel_path,
                        "label": int(label_str),
                        "speech_path": speech_path,
                        "dynamic_path": dynamic_path if osp.exists(dynamic_path) else None,
                    }
                )

        if len(self.samples) == 0:
            raise RuntimeError("No samples found for FeatureDataset. Check feature paths and file list.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        try:
            sample = self.samples[idx]
            speech_data = torch.load(sample["speech_path"], map_location="cpu")

            # Select semantic features based on use_concat flag
            if self.use_concat and "concat" in speech_data:
                # Use concatenated visual + audio features (2048-dim)
                sem_feat = _to_tensor(speech_data["concat"])
            elif self.use_concat:
                # Fallback: create concat if not available
                visual = _to_tensor(speech_data.get("visual"))
                audio = _to_tensor(speech_data.get("audio"))
                if visual.shape[0] != audio.shape[0]:
                    # Align before concat
                    audio = _resample_sequence(audio, visual.shape[0])
                sem_feat = torch.cat([visual, audio], dim=-1)
            else:
                # Use visual only (1024-dim)
                sem_feat = _to_tensor(speech_data.get("visual"))

            sem_feat_original_len = sem_feat.shape[0]
            sem_feat = _resample_sequence(sem_feat, self.seq_len)

            dynamic_data = None
            dyn_feat = None
            dyn_feat_original_len = 0
            if sample["dynamic_path"] and osp.exists(sample["dynamic_path"]):
                dynamic_data = torch.load(sample["dynamic_path"], map_location="cpu")
                temporal = dynamic_data.get("temporal")
                if temporal is not None:
                    dyn_feat_original_len = temporal.shape[0]
                    dyn_feat = _resample_sequence(_to_tensor(temporal), self.seq_len)

            if dyn_feat is None:
                if self.require_dynamic:
                    raise RuntimeError(f"Dynamic features missing for {sample['video']}")
                # DynamicForensics MS-TCN outputs 768-dim features
                dyn_feat = torch.zeros(self.seq_len, 768)

            audio_feat = None
            if self.include_audio and speech_data.get("audio") is not None:
                audio_feat = _resample_sequence(_to_tensor(speech_data["audio"]), self.seq_len)

            # Create padding mask if requested
            mask = None
            if self.create_mask and dyn_feat_original_len > 0:
                mask = self._create_mask(sem_feat_original_len, dyn_feat_original_len)

            result = {
                "video": sample["video"],
                "label": torch.tensor(sample["label"], dtype=torch.long),
                "sem_feat": sem_feat,
                "dyn_feat": dyn_feat,
                "speech_similarity": float(speech_data.get("similarity", 0.0)),
                "dynamic_logit": float(dynamic_data.get("avg_logit", 0.0)) if dynamic_data else 0.0,
            }

            # Only include audio_feat if it's not None
            if audio_feat is not None:
                result["audio_feat"] = audio_feat

            if mask is not None:
                result["mask"] = mask

            # Apply augmentation if enabled
            if self.augment:
                result = self._augment_features(result)

            return result

        except Exception as e:
            print(f"Error loading sample {idx} (video: {sample.get('video', 'unknown')}): {e}")
            raise

    def _create_mask(self, sem_len: int, dyn_len: int) -> torch.Tensor:
        """
        Create padding mask based on original feature lengths.

        Parameters
        ----------
        sem_len : int
            Original length of semantic features
        dyn_len : int
            Original length of dynamic features (number of clips)

        Returns
        -------
        torch.Tensor
            Boolean mask [seq_len], True indicates valid positions
        """
        # Compute effective frames (conservative: take minimum)
        # dyn_len is number of clips, each clip = 25 frames
        effective_frames = min(sem_len, dyn_len * 25)
        max_possible_frames = max(sem_len, dyn_len * 25)

        # Compute valid ratio in seq_len
        valid_ratio = effective_frames / max_possible_frames if max_possible_frames > 0 else 1.0
        valid_len = max(1, int(self.seq_len * valid_ratio))

        # Create mask
        mask = torch.zeros(self.seq_len, dtype=torch.bool)
        mask[:valid_len] = True

        return mask

    def _augment_features(self, sample: Dict) -> Dict:
        """
        Apply data augmentation to features.

        Augmentations:
        1. Temporal reversal (30% probability)
        2. Temporal shift (30% probability)
        3. Feature dropout (20% probability)
        """
        # 1. Temporal reversal
        if torch.rand(1) < 0.3:
            sample["sem_feat"] = torch.flip(sample["sem_feat"], dims=[0])
            sample["dyn_feat"] = torch.flip(sample["dyn_feat"], dims=[0])

        # 2. Temporal shift
        if torch.rand(1) < 0.3:
            shift = torch.randint(-self.seq_len // 4, self.seq_len // 4, (1,)).item()
            sample["sem_feat"] = torch.roll(sample["sem_feat"], shifts=shift, dims=0)
            sample["dyn_feat"] = torch.roll(sample["dyn_feat"], shifts=shift, dims=0)

        # 3. Feature dropout
        if torch.rand(1) < 0.2:
            dropout_prob = 0.1
            sem_mask = torch.rand_like(sample["sem_feat"]) > dropout_prob
            sample["sem_feat"] = sample["sem_feat"] * sem_mask

            dyn_mask = torch.rand_like(sample["dyn_feat"]) > dropout_prob
            sample["dyn_feat"] = sample["dyn_feat"] * dyn_mask

        return sample


def _safe_collate(batch):
    """Custom collate function that filters out None values."""
    from torch.utils.data.dataloader import default_collate

    # Filter out None values (failed samples)
    batch = [item for item in batch if item is not None]

    if len(batch) == 0:
        return None

    return default_collate(batch)


def create_dataloader(
    file_list: str,
    feature_root: str,
    batch_size: int = 8,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs,
) -> Tuple[FeatureDataset, DataLoader]:
    dataset = FeatureDataset(
        file_list=file_list,
        feature_root=feature_root,
        **dataset_kwargs,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=_safe_collate,
    )
    return dataset, loader
