"""
Inference script for generating predictions on test images.
"""
import os
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import get_enhanced_unet
from src.dataset import BUSITestDataset
from src.config import InferenceConfig, ModelConfig


def rle_encode_mask(mask):
    """
    Encode binary mask to Run-Length Encoding (RLE).
    
    Args:
        mask: Binary mask (H, W)
        
    Returns:
        RLE string
    """
    if np.sum(mask) == 0:
        return ''
    
    pixels = mask.flatten()
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = np.concatenate([[0], runs])
    run_lengths = []
    
    for i in range(len(runs) - 1):
        if pixels[runs[i]] == 1:
            start = runs[i] + 1
            length = runs[i + 1] - runs[i]
            run_lengths.extend([start, length])
    
    return ' '.join(str(x) for x in run_lengths)


def combined_encode(masks_dict, delimiter="~"):
    """
    Encode multiple class masks into a combined string.
    
    Args:
        masks_dict: Dictionary mapping class_id to binary mask
        delimiter: Delimiter between class encodings
        
    Returns:
        Combined encoded string
    """
    if not masks_dict:
        return ""
    
    encoded_parts = []
    for class_id, mask in masks_dict.items():
        rle = rle_encode_mask(mask)
        if rle:
            encoded_parts.append(f"{class_id}:{rle}")
    
    return delimiter.join(encoded_parts)


def generate_submission(model, test_dir, output_file, batch_size=16, 
                       device='cuda', image_size=(256, 256)):
    """
    Generate submission file for test images.
    
    Args:
        model: Trained model
        test_dir: Directory containing test images
        output_file: Output CSV file path
        batch_size: Batch size for inference
        device: Device to run inference on
        image_size: Target image size
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    
    # Create test dataset and loader
    test_dataset = BUSITestDataset(test_dir, image_size=image_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = model.to(device)
    model.eval()
    
    results = {'ID': [], 'encoded_pixels': []}
    
    print(f"Generating predictions for {len(test_dataset)} images...")
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Processing"):
            images = batch['image'].to(device)
            image_ids = batch['image_id']
            original_hs = batch['original_h']
            original_ws = batch['original_w']
            
            # Get predictions
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1).cpu().numpy()
            
            batch_size = images.shape[0]
            
            for i in range(batch_size):
                image_id = image_ids[i]
                pred = predictions[i]
                orig_h, orig_w = original_hs[i].item(), original_ws[i].item()
                
                # Resize prediction to original size
                pred_resized = cv2.resize(
                    pred.astype(np.float32), 
                    (orig_w, orig_h), 
                    interpolation=cv2.INTER_NEAREST
                ).astype(np.int32)
                
                # Create masks for each class (excluding background)
                masks_dict = {}
                for class_id in range(1, outputs.shape[1]):
                    binary_mask = (pred_resized == class_id).astype(np.uint8)
                    if np.sum(binary_mask) > 0:
                        masks_dict[class_id] = binary_mask
                
                # Encode masks
                encoded_pixels = combined_encode(masks_dict)
                results['ID'].append(image_id)
                results['encoded_pixels'].append(encoded_pixels)
    
    # Create DataFrame and save
    submission_df = pd.DataFrame(results)
    submission_df['encoded_pixels'] = submission_df['encoded_pixels'].fillna('')
    submission_df.loc[submission_df['encoded_pixels'] == '', 'encoded_pixels'] = '<empty>'
    submission_df = submission_df.sort_values('ID')
    submission_df.to_csv(output_file, index=False)
    
    print(f"\nSubmission saved to {output_file}")
    print(f"Total entries: {len(submission_df)}")
    print(f"Empty predictions: {(submission_df['encoded_pixels'] == '<empty>').sum()}")
    
    return submission_df


if __name__ == "__main__":
    # Configuration
    model_config = ModelConfig()
    inference_config = InferenceConfig()
    
    # Set device
    device = torch.device(inference_config.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = get_enhanced_unet(
        encoder_name=model_config.encoder_name,
        encoder_weights=None,  # Don't load ImageNet weights for inference
        in_channels=model_config.in_channels,
        out_channels=model_config.out_channels
    )
    
    # Load trained weights
    if os.path.exists(inference_config.model_path):
        model.load_state_dict(torch.load(inference_config.model_path, map_location=device))
        print(f"Loaded model from {inference_config.model_path}")
    else:
        raise FileNotFoundError(f"Model not found at {inference_config.model_path}")
    
    model = model.to(device)
    model.eval()
    
    # Generate submission
    generate_submission(
        model=model,
        test_dir=inference_config.test_dir,
        output_file=inference_config.output_file,
        batch_size=inference_config.batch_size,
        device=device,
        image_size=inference_config.image_size
    )




