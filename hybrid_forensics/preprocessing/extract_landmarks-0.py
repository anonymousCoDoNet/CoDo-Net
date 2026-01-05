"""
Extract facial landmarks from videos using face-alignment
Adapted from SpeechForensics preprocessing
"""

import cv2
import numpy as np
import torch
import json
import os
import os.path as osp
from tqdm import tqdm
import argparse
from pathlib import Path
import sys

# Try to import face_alignment
try:
    import face_alignment
    HAS_FACE_ALIGNMENT = True
except ImportError:
    HAS_FACE_ALIGNMENT = False


def extract_landmarks_from_video(video_path, fa, max_frames=None):
    """
    Extract 68-point facial landmarks from video frames
    
    Parameters
    ----------
    video_path : str
        Path to video file
    fa : face_alignment.FaceAlignment
        Face alignment detector
    max_frames : int, optional
        Maximum number of frames to process
    
    Returns
    -------
    dict
        Dictionary mapping frame names to landmarks
    """
    landmarks_dict = {}
    
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if max_frames is not None and frame_idx >= max_frames:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect landmarks
        try:
            landmark = fa.get_landmarks(frame_rgb)
            if landmark is not None and len(landmark) > 0:
                landmark = landmark[0].tolist()
            else:
                landmark = None
        except Exception as e:
            landmark = None
        
        frame_name = f'{frame_idx:04d}.jpg'
        landmarks_dict[frame_name] = landmark
        frame_idx += 1
    
    cap.release()
    return landmarks_dict


def extract_landmarks_main(args):
    """
    Main function to extract landmarks from video list
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing:
        - video_root: root directory of videos
        - file_list: path to file list
        - output_dir: output directory for landmarks
        - face_detector: path to face detector model
        - max_frames: maximum frames per video
    """
    print("="*60)
    print("Extracting Facial Landmarks")
    print("="*60)
    
    # Initialize face alignment detector
    print("\nInitializing face alignment detector...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Resolve face detector path
    from ..config import resolve_path
    face_detector_path = resolve_path(args.face_detector)
    
    print(f"Using face detector: {face_detector_path}")
    print(f"Device: {device}")
    
    fa = None
    
    # Method 1: Try using custom RetinaFace detector (from SpeechForensics)
    try:
        print("Attempting to load custom RetinaFace detector...")
        from ..detection import FaceDetector as CustomFaceDetector
        
        # Create a wrapper to make custom detector compatible with face-alignment API
        class CustomRetinaFaceWrapper:
            """Wrapper to use custom RetinaFace detector with face-alignment API"""
            def __init__(self, detector):
                self.detector = detector
            
            def get_landmarks(self, frame):
                """Get landmarks from frame using face-alignment"""
                # This will be handled by face-alignment library
                # We just need the detector to work with face-alignment's landmark detection
                return None
        
        detector = CustomFaceDetector(device=device, path_to_detector=face_detector_path, verbose=False)
        print("✓ Custom RetinaFace detector loaded!")
        
        # Now wrap it with face-alignment for landmark detection
        if HAS_FACE_ALIGNMENT:
            try:
                # Create a custom face-alignment instance that uses our detector
                fa = face_alignment.FaceAlignment(
                    face_alignment.LandmarksType.TWO_D,
                    device=device
                )
                # Replace the detector with our custom one
                fa.face_detector = detector
                print("✓ Using custom RetinaFace with face-alignment landmark detection!")
            except Exception as e:
                print(f"✗ Failed to integrate custom detector with face-alignment: {e}")
                fa = None
        else:
            print("⚠ face-alignment not available, using custom detector only")
            fa = CustomRetinaFaceWrapper(detector)
    except Exception as e:
        print(f"✗ Custom RetinaFace detector failed: {e}")
        fa = None
    
    # Method 2: Try using face-alignment library with retinaface
    if fa is None and HAS_FACE_ALIGNMENT:
        try:
            print("Attempting to load RetinaFace via face-alignment library...")
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                face_detector='retinaface',
                device=device,
                face_detector_kwargs={'path_to_detector': face_detector_path}
            )
            print("✓ Face detector (RetinaFace via face-alignment) loaded!")
        except (ModuleNotFoundError, Exception) as e:
            print(f"✗ RetinaFace via face-alignment failed: {e}")
            fa = None
    
    # Method 3: Fallback to dlib
    if fa is None and HAS_FACE_ALIGNMENT:
        try:
            print("Attempting to use dlib detector as fallback...")
            fa = face_alignment.FaceAlignment(
                face_alignment.LandmarksType.TWO_D,
                face_detector='dlib',
                device=device
            )
            print("✓ Face detector (dlib) loaded!")
        except Exception as e:
            print(f"✗ dlib detector failed: {e}")
            fa = None
    
    # If all methods failed, raise error
    if fa is None:
        print("\n" + "="*60)
        print("ERROR: Could not load any face detector!")
        print("="*60)
        print("\nPlease try one of the following:")
        print("1. Install face-alignment with RetinaFace:")
        print("   pip install face-alignment[pytorch]")
        print("\n2. Or copy RetinaFace from SpeechForensics:")
        print("   bash setup_retinaface_from_speechforensics.sh")
        print("\n3. Or install dlib:")
        print("   pip install dlib")
        raise RuntimeError("No face detector available")
    
    # Read video list
    with open(args.file_list, 'r') as f:
        video_list = f.read().split('\n')
    
    video_list = [v.strip() for v in video_list if v.strip()]
    print(f"\nFound {len(video_list)} videos to process")
    
    # Process each video
    processed = 0
    skipped = 0
    
    for video_item in tqdm(video_list, desc="Extracting landmarks"):
        video_rel_path = video_item.split(' ')[0]
        video_path = osp.join(args.video_root, video_rel_path)
        
        if not osp.exists(video_path):
            skipped += 1
            continue
        
        # Determine output path - create directory structure matching crop_mouths.py expectations
        # Expected structure: output_dir/video_name/landmarks.json
        video_rel_path_no_ext = osp.splitext(video_rel_path)[0]
        out_path = osp.join(args.output_dir, video_rel_path_no_ext, 'landmarks.json')
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        
        # Skip if already processed
        if osp.exists(out_path):
            processed += 1
            continue
        
        try:
            # Extract landmarks
            landmarks_dict = extract_landmarks_from_video(
                video_path, fa, max_frames=args.max_frames
            )
            
            # Save landmarks
            with open(out_path, 'w') as f:
                json.dump(landmarks_dict, f)
            
            processed += 1
        except Exception as e:
            print(f"  Error processing {video_path}: {e}")
            skipped += 1
    
    print("\n" + "="*60)
    print("Landmark Extraction Complete")
    print("="*60)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract facial landmarks from videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--video_root',
        type=str,
        required=True,
        help='Root directory of videos'
    )
    parser.add_argument(
        '--file_list',
        type=str,
        required=True,
        help='Path to file list (format: "video_path label")'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Output directory for landmarks'
    )
    parser.add_argument(
        '--face_detector',
        type=str,
        default='checkpoints/Resnet50_Final.pth',
        help='Path to face detector model'
    )
    parser.add_argument(
        '--max_frames',
        type=int,
        default=None,
        help='Maximum frames per video (None for all)'
    )
    
    args = parser.parse_args()
    extract_landmarks_main(args)
