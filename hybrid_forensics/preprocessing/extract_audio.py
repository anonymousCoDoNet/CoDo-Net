"""
Extract audio from videos and convert to WAV format
"""

import os
import os.path as osp
import subprocess
import cv2
from tqdm import tqdm
import argparse


def extract_audio_from_video(video_path, output_path, ffmpeg_path='/usr/bin/ffmpeg', sample_rate=16000):
    """
    Extract audio from video and save as WAV
    
    Parameters
    ----------
    video_path : str
        Path to video file
    output_path : str
        Path to output WAV file
    ffmpeg_path : str
        Path to ffmpeg executable
    sample_rate : int
        Sample rate for audio (default: 16000 Hz)
    
    Returns
    -------
    bool
        True if successful, False otherwise
    """
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    
    # Remove existing file if present
    if osp.exists(output_path):
        os.remove(output_path)
    
    # Extract audio using ffmpeg
    cmd = f"{ffmpeg_path} -i {video_path} -f wav -vn -ar {sample_rate} -ac 1 -y {output_path} -loglevel quiet"
    
    try:
        result = subprocess.call(cmd, shell=True)
        return result == 0 and osp.exists(output_path)
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return False


def extract_audio_main(args):
    """
    Main function to extract audio from video list
    
    Parameters
    ----------
    args : argparse.Namespace
        Arguments containing:
        - video_root: root directory of videos
        - file_list: path to file list
        - output_dir: output directory for audio
        - ffmpeg: path to ffmpeg executable
        - sample_rate: audio sample rate
    """
    print("="*60)
    print("Extracting Audio from Videos")
    print("="*60)
    print(f"Video root: {args.video_root}")
    print(f"Output directory: {args.output_dir}")
    print(f"Sample rate: {args.sample_rate} Hz")
    print(f"FFmpeg path: {args.ffmpeg}")
    
    # Read video list
    with open(args.file_list, 'r') as f:
        video_list = f.read().split('\n')
    
    video_list = [v.strip() for v in video_list if v.strip()]
    print(f"\nFound {len(video_list)} videos to process")
    
    # Process each video
    processed = 0
    skipped = 0
    failed = 0
    
    for video_item in tqdm(video_list, desc="Extracting audio"):
        video_rel_path = video_item.split(' ')[0]
        video_path = osp.join(args.video_root, video_rel_path)
        
        if not osp.exists(video_path):
            skipped += 1
            continue
        
        # Determine output path - preserve directory structure
        video_rel_path_no_ext = osp.splitext(video_rel_path)[0]
        out_path = osp.join(args.output_dir, video_rel_path_no_ext + '.wav')
        os.makedirs(osp.dirname(out_path), exist_ok=True)
        
        # Skip if already processed
        if osp.exists(out_path):
            processed += 1
            continue
        
        # Extract audio
        success = extract_audio_from_video(
            video_path, out_path, 
            ffmpeg_path=args.ffmpeg,
            sample_rate=args.sample_rate
        )
        
        if success:
            processed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print("Audio Extraction Complete")
    print("="*60)
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")
    print(f"Output directory: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract audio from videos',
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
        help='Output directory for audio files'
    )
    parser.add_argument(
        '--ffmpeg',
        type=str,
        default='/usr/bin/ffmpeg',
        help='Path to ffmpeg executable'
    )
    parser.add_argument(
        '--sample_rate',
        type=int,
        default=16000,
        help='Audio sample rate in Hz'
    )
    
    args = parser.parse_args()
    extract_audio_main(args)
