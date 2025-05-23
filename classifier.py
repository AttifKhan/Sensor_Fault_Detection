from pathlib import Path
from typing import List, Tuple
import logging

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from src.categories import DocumentCategories
from src.handler import FileHandler
from src.organizer import DocumentOrganizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CLIPClassifier:
    """CLIP-based document classifier"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", cache_dir: str = "./models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model cache directory: {self.cache_dir}")
        
        # Create cache directory if it doesn't exist
        if not self.cache_dir.exists():
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created cache directory: {self.cache_dir}")
        
        # Use the model name as is for the local path (transformers handles path normalization)
        model_path = self.cache_dir / model_name.replace("/", "_")
        
        try:
            if self._model_exists_locally(model_path):
                logger.info(f"Loading cached model from: {model_path}")
                self.model = CLIPModel.from_pretrained(str(model_path)).to(self.device)
                # Removed use_fast=True to fix the error
                self.processor = CLIPProcessor.from_pretrained(str(model_path))
                logger.info("CLIP model loaded from local cache successfully")
            else:
                logger.info(f"Model not found locally. Downloading {model_name}...")
                
                # Download and cache model
                self.model = CLIPModel.from_pretrained(model_name).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name)
                
                # Save model locally for future use
                logger.info(f"Saving model to local cache: {model_path}")
                if not model_path.exists():
                    model_path.mkdir(parents=True, exist_ok=True)
                    logger.info(f"Created model directory: {model_path}")
                
                self.model.save_pretrained(str(model_path))
                self.processor.save_pretrained(str(model_path))
                logger.info("CLIP model downloaded and cached successfully")
                
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def _model_exists_locally(self, model_path: Path) -> bool:
        """Check if model files exist locally"""
        # Updated list of required files for CLIP models
        required_files = [
            "config.json",
            "preprocessor_config.json",
        ]
        
        # CLIP models can have different weight file formats
        weight_files = [
            "pytorch_model.bin",
            "model.safetensors"  # Newer format
        ]
        
        # Tokenizer files (may vary)
        tokenizer_files = [
            "tokenizer_config.json",
            "vocab.json",
            "merges.txt"  # For BPE tokenizers
        ]
        
        if not model_path.exists():
            logger.debug(f"Model directory does not exist: {model_path}")
            return False
        
        # Check required config files
        for file_name in required_files:
            if not (model_path / file_name).exists():
                logger.debug(f"Missing required file: {file_name}")
                return False
        
        # Check if at least one weight file exists
        has_weights = any((model_path / weight_file).exists() for weight_file in weight_files)
        if not has_weights:
            logger.debug("No weight files found (pytorch_model.bin or model.safetensors)")
            return False
        
        # Check if tokenizer files exist (at least tokenizer_config.json should be present)
        if not (model_path / "tokenizer_config.json").exists():
            logger.debug("Missing tokenizer_config.json")
            return False
                
        logger.debug("All required model files found locally")
        return True
    
    def classify_images(self, images: List[Image.Image]) -> Tuple[str, float]:
        """Classify images using CLIP model"""
        try:
            # Prepare all category prompts
            all_prompts = []
            category_mapping = {}
            
            for category, prompts in DocumentCategories.CATEGORIES.items():
                for prompt in prompts:
                    all_prompts.append(prompt)
                    category_mapping[prompt] = category
            
            # Get predictions for all images
            image_scores = []
            
            for image in images:
                # Process image and text
                inputs = self.processor(
                    text=all_prompts,
                    images=image,
                    return_tensors="pt",
                    padding=True
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits_per_image = outputs.logits_per_image
                    temperature = 0.5
                    probs = (logits_per_image/temperature).softmax(dim=1)
                    
                image_scores.append(probs.cpu().numpy()[0])
            
    
            avg_scores = torch.tensor(image_scores).mean(dim=0)
            
            # Get best matching prompt and its category
            best_prompt_idx = avg_scores.argmax().item()
            best_prompt = all_prompts[best_prompt_idx]
            predicted_category = category_mapping[best_prompt]
            confidence = avg_scores[best_prompt_idx].item()
            
            logger.info(f"Classified as: {predicted_category} with confidence: {confidence:.3f}")
            return predicted_category, confidence
            
        except Exception as e:
            logger.error(f"Error during classification: {e}")
            return "others", 0.0


class DocumentClassificationService:
    """Main service class combining all components"""
    
    def __init__(self, model_cache_dir: str = "./models"):
        self.classifier = CLIPClassifier(cache_dir=model_cache_dir)
        self.organizer = DocumentOrganizer()
        self.file_handler = FileHandler()
    
    async def process_document(self, file_path: str, 
                             original_filename: str) -> dict:
        """Process a single document: classify and organize"""
        try:
            # Convert document to images
            images = self.file_handler.convert_to_images(file_path)
            
            if not images:
                raise ValueError("Could not extract images from document")
            
            # Classify document
            category, confidence = self.classifier.classify_images(images)
            
            # Save document in appropriate directory
            saved_path = self.organizer.save_document(
                images, category, original_filename
            )
            
            return {
                "status": "success",
                "original_filename": original_filename,
                "category": category,
                "confidence": float(confidence),
                "saved_path": saved_path,
                "pages_processed": len(images)
            }
            
        except Exception as e:
            logger.error(f"Error processing document {original_filename}: {e}")
            return {
                "status": "error",
                "original_filename": original_filename,
                "error": str(e)
            }