#!/usr/bin/env python3
"""
FaceForensics++ Download Script v2
Based on official FaceForensics++ repository and email instructions.

Usage:
    python download_FaceForensics_v2.py <output_path> -d all -c c23 -t videos --server EU2
    python download_FaceForensics_v2.py <output_path> -d original -c raw -t videos --server EU2
    python download_FaceForensics_v2.py <output_path> -d DeepFakeDetection -c raw -t videos --server EU2
"""

import argparse
import os
import sys
import urllib.request
import urllib.error
import tempfile
from pathlib import Path
from tqdm import tqdm

# Server configuration - based on email instructions
SERVERS = {
    'EU': 'http://kaldir.vc.in.tum.de/FaceForensics/',
    'EU2': 'http://kaldir.vc.in.tum.de/FaceForensics/',  # EU2 redirects to EU
    'CA': 'http://kaldir.vc.in.tum.de/FaceForensics/'   # CA server (if available)
}

# Dataset types
DATASET_TYPES = [
    'all', 'original', 'DeepFakeDetection', 'DeepFakeDetection_original',
    'Face2Face', 'FaceSwap', 'Deepfakes', 'NeuralTextures'
]

# Compression types
COMPRESSION_TYPES = ['raw', 'c0', 'c23', 'c40']

# Content types
CONTENT_TYPES = ['videos', 'masks', 'images']

def test_server_connection(server_url):
    """Test if server is accessible."""
    try:
        print(f"Testing connection to {server_url}...")
        response = urllib.request.urlopen(server_url, timeout=10)
        print(f"Server accessible! Status: {response.getcode()}")
        return True
    except Exception as e:
        print(f"Server connection failed: {e}")
        return False

def download_file(url, output_path, desc=None):
    """Download a file with progress bar."""
    try:
        print(f"Downloading: {url}")
        print(f"Output: {output_path}")
        
        with tqdm(unit='B', unit_scale=True, desc=desc) as pbar:
            def progress_hook(count, block_size, total_size):
                if total_size > 0:
                    pbar.total = total_size
                    pbar.update(block_size)
            
            urllib.request.urlretrieve(url, output_path, progress_hook)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def get_file_list(server_url, dataset_type, compression, content_type):
    """Get the list of files to download."""
    # Try different URL patterns based on documentation
    possible_urls = [
        f"{server_url}filelists/{dataset_type}_{compression}_{content_type}.txt",
        f"{server_url}v2/filelists/{dataset_type}_{compression}_{content_type}.txt",
        f"{server_url}filelists/{dataset_type}.txt",
        f"{server_url}v2/filelists/{dataset_type}.txt"
    ]
    
    for url in possible_urls:
        try:
            print(f"Trying filelist URL: {url}")
            with urllib.request.urlopen(url, timeout=10) as response:
                files = [line.decode('utf-8').strip() for line in response if line.strip()]
                if files:
                    print(f"Found {len(files)} files in {url}")
                    return files
        except urllib.error.HTTPError as e:
            print(f"HTTP Error {e.code} for {url}")
        except Exception as e:
            print(f"Error accessing {url}: {e}")
    
    print("No file list found with any URL pattern")
    return []

def main():
    parser = argparse.ArgumentParser(description='Download FaceForensics++ dataset')
    parser.add_argument('output_path', help='Output directory for downloaded files')
    parser.add_argument('-d', '--dataset', default='all', choices=DATASET_TYPES,
                       help='Dataset type to download')
    parser.add_argument('-c', '--compression', default='c23', choices=COMPRESSION_TYPES,
                       help='Compression type')
    parser.add_argument('-t', '--type', default='videos', choices=CONTENT_TYPES,
                       help='Content type to download')
    parser.add_argument('--server', default='EU2', choices=['EU', 'EU2', 'CA'],
                       help='Server to download from')
    parser.add_argument('--num_videos', type=int, default=None,
                       help='Number of videos to download (for testing)')
    parser.add_argument('--test-only', action='store_true',
                       help='Only test server connection and file list')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("FaceForensics++ Download Script v2")
    print("=" * 60)
    
    # Validate arguments
    if args.dataset not in DATASET_TYPES:
        print(f"Invalid dataset type: {args.dataset}")
        print(f"Available types: {DATASET_TYPES}")
        sys.exit(1)
    
    if args.compression not in COMPRESSION_TYPES:
        print(f"Invalid compression type: {args.compression}")
        print(f"Available types: {COMPRESSION_TYPES}")
        sys.exit(1)
    
    if args.type not in CONTENT_TYPES:
        print(f"Invalid content type: {args.type}")
        print(f"Available types: {CONTENT_TYPES}")
        sys.exit(1)
    
    # Setup paths
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get server URL
    server_url = SERVERS[args.server]
    print(f"Using server: {args.server} ({server_url})")
    
    # Test server connection
    if not test_server_connection(server_url):
        print("Server connection failed. Please check your internet connection.")
        sys.exit(1)
    
    # Get file list
    print(f"\nGetting file list for {args.dataset}_{args.compression}_{args.type}...")
    files = get_file_list(server_url, args.dataset, args.compression, args.type)
    
    if not files:
        print("No files found to download!")
        print("\nPossible reasons:")
        print("1. You need to apply for dataset access first")
        print("2. The server URLs have changed")
        print("3. The dataset structure is different")
        sys.exit(1)
    
    if args.test_only:
        print(f"Test mode: Found {len(files)} files")
        print("First 5 files:")
        for i, f in enumerate(files[:5]):
            print(f"  {i+1}. {f}")
        return
    
    # Limit files if num_videos is specified
    if args.num_videos:
        files = files[:args.num_videos]
        print(f"Limiting download to {args.num_videos} files")
    
    print(f"Found {len(files)} files to download")
    
    # Download files
    base_url = f"{server_url}{args.dataset}/{args.compression}/{args.type}/"
    print(f"Base URL: {base_url}")
    
    successful_downloads = 0
    for i, filename in enumerate(files):
        file_url = base_url + filename
        output_file = output_dir / filename
        
        # Create subdirectories if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        print(f"\nDownloading {i+1}/{len(files)}: {filename}")
        success = download_file(file_url, output_file, desc=filename)
        
        if success:
            successful_downloads += 1
        else:
            print(f"Failed to download {filename}")
            continue
    
    print(f"\nDownload completed!")
    print(f"Successfully downloaded: {successful_downloads}/{len(files)} files")
    print(f"Files saved to: {output_dir}")

if __name__ == "__main__":
    main() 