"""
Feature alignment utilities for CoDo-Net.

This module handles temporal and channel dimension alignment between
SpeechForensics (frame-level, 1024-dim) and LipForensics (clip-level, 512-dim).
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional


def align_temporal_features(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor],
    target_len: int = 64,
    use_concat: bool = True
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Align SpeechForensics and LipForensics features to the same temporal length.

    Key Challenge:
    - SpeechForensics: [T_frames, 1024], where T ≈ fps × duration (e.g., 25×4 = 100)
    - LipForensics: [N_clips, 512], where N = T_frames / 25 (e.g., 100/25 = 4)

    Solution:
    Use adaptive average pooling to resample both sequences to target_len.

    Parameters
    ----------
    speech_feat : dict
        SpeechForensics feature dictionary containing:
        - 'visual': torch.Tensor, shape [T, 1024]
        - 'audio': torch.Tensor, shape [T, 1024]
        - 'concat': torch.Tensor, shape [T, 2048] (visual + audio concatenated)
    lip_feat : dict
        LipForensics feature dictionary containing:
        - 'temporal': torch.Tensor, shape [N, 512]
    target_len : int
        Target temporal length for alignment (default: 64)
    use_concat : bool
        If True, use concat features (2048-dim); otherwise use visual only (1024-dim)

    Returns
    -------
    sem_aligned : torch.Tensor
        Aligned semantic features, shape [target_len, 2048] or [target_len, 1024]
    dyn_aligned : torch.Tensor
        Aligned dynamic features, shape [target_len, 512]

    Examples
    --------
    >>> speech_feat = {
    ...     'visual': torch.randn(100, 1024),
    ...     'audio': torch.randn(100, 1024),
    ...     'concat': torch.randn(100, 2048)
    ... }
    >>> lip_feat = {'temporal': torch.randn(4, 512)}
    >>> sem, dyn = align_temporal_features(speech_feat, lip_feat, target_len=64)
    >>> sem.shape, dyn.shape
    (torch.Size([64, 2048]), torch.Size([64, 512]))
    """
    # 1. Extract raw features
    if use_concat:
        if 'concat' not in speech_feat:
            # Fallback: create concat if not provided
            sem_raw = torch.cat([speech_feat['visual'], speech_feat['audio']], dim=-1)
        else:
            sem_raw = speech_feat['concat']  # [T, 2048]
    else:
        sem_raw = speech_feat['visual']  # [T, 1024]

    dyn_raw = lip_feat['temporal']  # [N, 512]

    # 2. Convert to [Batch=1, Channels, Time] for 1D pooling
    sem_T = sem_raw.transpose(0, 1).unsqueeze(0)  # [1, C_sem, T]
    dyn_T = dyn_raw.transpose(0, 1).unsqueeze(0)  # [1, 512, N]

    # 3. Adaptive average pooling to target_len
    sem_pooled = F.adaptive_avg_pool1d(sem_T, output_size=target_len)  # [1, C_sem, target_len]
    dyn_pooled = F.adaptive_avg_pool1d(dyn_T, output_size=target_len)  # [1, 512, target_len]

    # 4. Convert back to [Time, Channels]
    sem_aligned = sem_pooled.squeeze(0).transpose(0, 1)  # [target_len, C_sem]
    dyn_aligned = dyn_pooled.squeeze(0).transpose(0, 1)  # [target_len, 512]

    return sem_aligned, dyn_aligned


def create_padding_mask(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor],
    target_len: int = 64,
    frames_per_clip: int = 25
) -> torch.Tensor:
    """
    Create a padding mask for variable-length sequences.

    The mask indicates which temporal positions are valid (True) vs padded (False).
    This is useful when videos have different lengths but are batched together.

    Parameters
    ----------
    speech_feat : dict
        SpeechForensics features with 'visual' or 'concat' key
    lip_feat : dict
        LipForensics features with 'temporal' key
    target_len : int
        Target temporal length after alignment
    frames_per_clip : int
        Number of frames per clip in LipForensics (default: 25)

    Returns
    -------
    mask : torch.Tensor
        Boolean mask, shape [target_len], True indicates valid positions

    Notes
    -----
    The effective length is computed as the minimum of:
    - Total frames from SpeechForensics (T)
    - Total frames from LipForensics (N × frames_per_clip)

    This conservative approach ensures we don't use positions where one model
    might not have valid features.
    """
    # Get original lengths
    if 'concat' in speech_feat:
        T = speech_feat['concat'].shape[0]
    else:
        T = speech_feat['visual'].shape[0]

    N = lip_feat['temporal'].shape[0]

    # Compute effective frames (conservative: take minimum)
    effective_frames = min(T, N * frames_per_clip)
    max_possible_frames = max(T, N * frames_per_clip)

    # Compute valid ratio in target_len
    valid_ratio = effective_frames / max_possible_frames if max_possible_frames > 0 else 1.0
    valid_len = max(1, int(target_len * valid_ratio))  # At least 1 valid position

    # Create mask
    mask = torch.zeros(target_len, dtype=torch.bool)
    mask[:valid_len] = True

    return mask


def batch_align_features(
    speech_feats: list,
    lip_feats: list,
    target_len: int = 64,
    use_concat: bool = True,
    create_mask: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """
    Batch version of align_temporal_features for DataLoader.

    Parameters
    ----------
    speech_feats : list of dict
        List of SpeechForensics feature dictionaries
    lip_feats : list of dict
        List of LipForensics feature dictionaries
    target_len : int
        Target temporal length
    use_concat : bool
        Whether to use concat features
    create_mask : bool
        Whether to create padding masks

    Returns
    -------
    sem_batch : torch.Tensor
        Batched semantic features, shape [B, target_len, C_sem]
    dyn_batch : torch.Tensor
        Batched dynamic features, shape [B, target_len, 512]
    mask_batch : torch.Tensor or None
        Batched masks, shape [B, target_len] if create_mask=True, else None
    """
    batch_size = len(speech_feats)
    sem_list = []
    dyn_list = []
    mask_list = []

    for i in range(batch_size):
        sem, dyn = align_temporal_features(
            speech_feats[i],
            lip_feats[i],
            target_len=target_len,
            use_concat=use_concat
        )
        sem_list.append(sem)
        dyn_list.append(dyn)

        if create_mask:
            mask = create_padding_mask(speech_feats[i], lip_feats[i], target_len)
            mask_list.append(mask)

    sem_batch = torch.stack(sem_list, dim=0)  # [B, target_len, C_sem]
    dyn_batch = torch.stack(dyn_list, dim=0)  # [B, target_len, 512]
    mask_batch = torch.stack(mask_list, dim=0) if create_mask else None

    return sem_batch, dyn_batch, mask_batch


def compute_temporal_statistics(
    speech_feat: Dict[str, torch.Tensor],
    lip_feat: Dict[str, torch.Tensor]
) -> Dict[str, float]:
    """
    Compute statistics about temporal misalignment.

    Useful for debugging and understanding feature extraction quality.

    Parameters
    ----------
    speech_feat : dict
        SpeechForensics features
    lip_feat : dict
        LipForensics features

    Returns
    -------
    stats : dict
        Dictionary containing:
        - 'speech_frames': Number of frames in SpeechForensics
        - 'lip_clips': Number of clips in LipForensics
        - 'temporal_ratio': Ratio of speech_frames to (lip_clips × 25)
        - 'alignment_quality': Quality score (1.0 = perfect, <1.0 = misaligned)
    """
    T = speech_feat.get('concat', speech_feat['visual']).shape[0]
    N = lip_feat['temporal'].shape[0]

    temporal_ratio = T / (N * 25) if N > 0 else 0.0
    # Ideal ratio should be close to 1.0
    alignment_quality = 1.0 - abs(1.0 - temporal_ratio)

    return {
        'speech_frames': int(T),
        'lip_clips': int(N),
        'temporal_ratio': float(temporal_ratio),
        'alignment_quality': float(alignment_quality)
    }


if __name__ == '__main__':
    # Test the alignment functions
    print("Testing feature alignment...")

    # Simulate features
    speech_feat = {
        'visual': torch.randn(100, 1024),
        'audio': torch.randn(100, 1024),
        'concat': torch.randn(100, 2048)
    }
    lip_feat = {
        'temporal': torch.randn(4, 512)
    }

    # Test alignment
    sem, dyn = align_temporal_features(speech_feat, lip_feat, target_len=64)
    print(f"✓ Aligned shapes: sem={sem.shape}, dyn={dyn.shape}")

    # Test mask creation
    mask = create_padding_mask(speech_feat, lip_feat, target_len=64)
    print(f"✓ Mask shape: {mask.shape}, valid positions: {mask.sum().item()}/{len(mask)}")

    # Test statistics
    stats = compute_temporal_statistics(speech_feat, lip_feat)
    print(f"✓ Temporal statistics: {stats}")

    # Test batch alignment
    batch_sem, batch_dyn, batch_mask = batch_align_features(
        [speech_feat, speech_feat],
        [lip_feat, lip_feat],
        target_len=64
    )
    print(f"✓ Batch shapes: sem={batch_sem.shape}, dyn={batch_dyn.shape}, mask={batch_mask.shape}")

    print("\nAll tests passed!")
