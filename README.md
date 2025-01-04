# Multi-stream Video Violence Detection 

## Project Overview
This project implements a multi-stream approach for violence detection in videos using YOLO object detection and TSN (Temporal Segment Networks). The system combines spatial information from YOLO's bounding boxes with temporal features from TSN to create a robust violence detection pipeline.

## Requirements
- Python 3.8+
- PyTorch
- MMAction2
- YOLO (Ultralytics)
- OpenCV
- Decord
- NumPy
- tqdm

## Installation
```bash
# Install MMAction2 and dependencies
pip install -U openmim
mim install mmengine
mim install mmcv>=2.0.0
mim install mmdet
mim install mmpose

# Install other requirements
pip install torch opencv-python decord numpy tqdm ultralytics
```

## Project Structure
```
project/
├── configs/
│   └── tsn_config.py
├── data/
│   ├── Violence/
│   └── NonViolence/
├── outputs/
│   ├── train/
│   │   ├── violence/
│   │   └── nonviolence/
│   └── val/
│       ├── violence/
│       └── nonviolence/
└── scripts/
    └── multistream_processing.py
```

## Features
- Dual-stream processing combining YOLO and TSN
- Frame-level feature extraction
- Position-aware encoding using YOLO bounding boxes
- Automatic train/validation split
- Batch processing with progress tracking
- Error handling and validation
- CUDA acceleration support

## Processing Pipeline
1. Video Frame Extraction
   - Reads frames at specified intervals
   - Converts color space and normalizes

2. YOLO Detection
   - Detects objects in each frame
   - Extracts bounding box coordinates
   - Normalizes position information

3. TSN Feature Extraction
   - Processes full frames
   - Extracts features from detected regions
   - Combines global and local features

4. Feature Storage
   - Saves combined features as NPZ files
   - Organizes by dataset split and class


## Output Format
Each processed video generates an NPZ file containing:
- full_frame_features: Global frame features from TSN
- yolo_positions: Normalized bounding box coordinates
- clipped_features: Local features from detected regions
- frame_indices: Processed frame numbers
- total_frames: Total number of frames in video

## Parameter Configuration
- frame_interval: Controls frame sampling rate (default: 5)
- val_ratio: Train/validation split ratio (default: 0.2)
- padding: Padding for detected regions (default: 20 pixels)

## Model Configuration
- TSN Model: Using MMAction2's TSN implementation
- YOLO Model: Using Ultralytics YOLOv8
- Input Size: 224x224 pixels
- Normalization: ImageNet statistics

## Error Handling
- Validates video files before processing
- Skips corrupted frames
- Reports processing errors
- Continues processing on failure

## Performance Considerations
- Uses CUDA when available
- Batch processing for efficiency
- Progress tracking for long operations
- Memory-efficient feature storage

## Limitations
- Requires GPU for optimal performance
- Memory usage scales with video resolution
- Processing time depends on frame count
- Required disk space for feature storage

## Future Improvements
- Multi-GPU support
- Real-time processing option
- Additional feature extractors
- Data augmentation pipeline
- Memory optimization
- Parallel processing enhancement

## Contact & Support
For issues and suggestions:
- Open an issue in the repository
- Document the error and steps to reproduce
- Include system specifications
- Attach relevant error logs

## License
MIT LICENSE

