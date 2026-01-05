"""
Command-line interface for Hybrid Forensics
"""

import argparse
import sys
from hybrid_forensics.preprocessing.extract_landmarks import extract_landmarks_main
from hybrid_forensics.preprocessing.crop_mouths import crop_mouths_main
from hybrid_forensics.preprocessing.extract_audio import extract_audio_main
from hybrid_forensics.inference.detect import detect_main
from hybrid_forensics.dataset_config import get_dataset_config


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Hybrid Forensics: Deepfake Detection Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract landmarks
  python -m hybrid_forensics preprocess landmarks --video_root data/videos --file_list data/list.txt --output_dir data/landmarks

  # Crop mouths
  python -m hybrid_forensics preprocess crop --video_root data/videos --landmarks_dir data/landmarks --output_dir data/mouths

  # Extract audio
  python -m hybrid_forensics preprocess audio --video_root data/videos --file_list data/list.txt --output_dir data/audio

  # Run detection
  python -m hybrid_forensics detect --file_list data/list.txt --speech_mouth_dir data/mouths --speech_audio_dir data/audio --dynamic_mouth_dir data/mouths --output_dir results/
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Preprocessing subcommand
    preprocess_parser = subparsers.add_parser('preprocess', help='Preprocessing commands')
    preprocess_subparsers = preprocess_parser.add_subparsers(dest='preprocess_cmd', help='Preprocessing step')
    
    # Landmarks extraction
    landmarks_parser = preprocess_subparsers.add_parser('landmarks', help='Extract facial landmarks')
    landmarks_parser.add_argument('--video_root', type=str, required=True, help='Root directory of videos')
    landmarks_parser.add_argument('--file_list', type=str, required=True, help='Path to file list')
    landmarks_parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    landmarks_parser.add_argument('--face_detector', type=str, default='checkpoints/Resnet50_Final.pth')
    landmarks_parser.add_argument('--max_frames', type=int, default=None)
    
    # Mouth cropping
    crop_parser = preprocess_subparsers.add_parser('crop', help='Crop mouth regions')
    crop_parser.add_argument('--video_root', type=str, required=True)
    crop_parser.add_argument('--landmarks_dir', type=str, required=True)
    crop_parser.add_argument('--output_dir', type=str, required=True)
    crop_parser.add_argument('--mean_face', type=str, default='data/20words_mean_face.npy')
    crop_parser.add_argument('--crop_width', type=int, default=96)
    crop_parser.add_argument('--crop_height', type=int, default=96)
    crop_parser.add_argument('--start_idx', type=int, default=48)
    crop_parser.add_argument('--stop_idx', type=int, default=68)
    crop_parser.add_argument('--window_margin', type=int, default=12)
    crop_parser.add_argument('--num_workers', type=int, default=8)
    crop_parser.add_argument('--skip_existing', action='store_true')
    
    # Audio extraction
    audio_parser = preprocess_subparsers.add_parser('audio', help='Extract audio')
    audio_parser.add_argument('--video_root', type=str, required=True)
    audio_parser.add_argument('--file_list', type=str, required=True)
    audio_parser.add_argument('--output_dir', type=str, required=True)
    audio_parser.add_argument('--ffmpeg', type=str, default='/usr/bin/ffmpeg')
    audio_parser.add_argument('--sample_rate', type=int, default=16000)
    
    # Detection subcommand
    detect_parser = subparsers.add_parser('detect', help='Run hybrid cascade detection')
    detect_parser.add_argument('--file_list', type=str, required=True,
                              help='Path to file list (format: "video_path label")')
    detect_parser.add_argument('--speech_mouth_dir', type=str, required=True,
                              help='Root directory containing cropped mouths (contains speech_mouth.mp4 files)')
    detect_parser.add_argument('--speech_audio_dir', type=str, required=True,
                              help='Directory containing audio files (.wav)')
    detect_parser.add_argument('--dynamic_mouth_dir', type=str, required=True,
                              help='Root directory containing cropped mouths (contains dynamic_mouth/ subdirectories)')
    detect_parser.add_argument('--output_dir', type=str, default='results/')
    detect_parser.add_argument('--dataset', type=str, default=None,
                              help='Dataset name (lrs2, lrw, fakeavceleb, avlips) - auto-loads thresholds')
    detect_parser.add_argument('--speech_threshold', type=float, default=None,
                              help='Override speech threshold (if not set, uses dataset default)')
    detect_parser.add_argument('--dynamic_threshold', type=float, default=None,
                              help='Override dynamic threshold (if not set, uses dataset default)')
    detect_parser.add_argument('--checkpoint_path', type=str, default='checkpoints/av_hubert/large_vox_iter5.pt')
    detect_parser.add_argument('--dynamic_model_path', type=str, default='checkpoints/dynamic/dynamic_ff.pth')
    detect_parser.add_argument('--max_length', type=int, default=50)
    
    # Dataset info subcommand
    dataset_parser = subparsers.add_parser('dataset', help='Dataset information and thresholds')
    dataset_subparsers = dataset_parser.add_subparsers(dest='dataset_cmd', help='Dataset command')
    
    # List datasets
    list_parser = dataset_subparsers.add_parser('list', help='List all available datasets')
    
    # Show dataset info
    info_parser = dataset_subparsers.add_parser('info', help='Show dataset information')
    info_parser.add_argument('--name', type=str, required=True, help='Dataset name')
    
    # Show guidelines
    guide_parser = dataset_subparsers.add_parser('guide', help='Show threshold tuning guidelines')
    
    args = parser.parse_args()
    
    if args.command == 'preprocess':
        if args.preprocess_cmd == 'landmarks':
            extract_landmarks_main(args)
        elif args.preprocess_cmd == 'crop':
            crop_mouths_main(args)
        elif args.preprocess_cmd == 'audio':
            extract_audio_main(args)
        else:
            preprocess_parser.print_help()
    elif args.command == 'detect':
        # Handle dataset-specific thresholds
        if args.dataset:
            dataset_config = get_dataset_config()
            thresholds = dataset_config.get_thresholds(args.dataset)
            
            # Use dataset thresholds if not overridden
            if args.speech_threshold is None:
                args.speech_threshold = thresholds['speech_threshold']
            if args.dynamic_threshold is None:
                args.dynamic_threshold = thresholds['dynamic_threshold']
            
            print(f"Using thresholds for dataset: {args.dataset}")
            print(f"  Speech threshold: {args.speech_threshold}")
            print(f"  Dynamic threshold: {args.dynamic_threshold}\n")
        else:
            # Use defaults if no dataset specified
            if args.speech_threshold is None:
                args.speech_threshold = 0.275
            if args.dynamic_threshold is None:
                args.dynamic_threshold = -8.92
        
        detect_main(args)
    elif args.command == 'dataset':
        dataset_config = get_dataset_config()
        
        if args.dataset_cmd == 'list':
            dataset_config.print_all_datasets()
        elif args.dataset_cmd == 'info':
            try:
                info = dataset_config.get_dataset_info(args.name)
                print(f"\n{'='*70}")
                print(f"DATASET: {args.name.upper()}")
                print(f"{'='*70}")
                for key, value in info.items():
                    print(f"  {key}: {value}")
                
                # Print metrics if available
                metrics = dataset_config.get_metrics(args.name)
                if metrics:
                    print(f"\n  Performance Metrics:")
                    for metric_name, metric_value in metrics.items():
                        print(f"    {metric_name}: {metric_value}")
                print(f"{'='*70}\n")
            except ValueError as e:
                print(f"Error: {e}")
        elif args.dataset_cmd == 'guide':
            dataset_config.print_guidelines()
        else:
            dataset_parser.print_help()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
