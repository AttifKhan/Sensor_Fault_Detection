import logging
from pathlib import Path
from typing import List
from PIL import Image

from src.categories import DocumentCategories
from src.handler import FileHandler


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentOrganizer:
    """Organize classified documents into directories"""
    
    def __init__(self, base_output_dir: str = "classified_documents"):
        self.base_output_dir = Path(base_output_dir)
        self.create_directories()
    
    def create_directories(self):
        """Create directories for each document category"""
        for category in DocumentCategories.CATEGORIES.keys():
            category_dir = self.base_output_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {category_dir}")
    
    def save_document(self, images: List[Image.Image], category: str, 
                     original_filename: str) -> str:
        """Save classified document as PDF in appropriate directory"""
        try:
            # Generate output filename
            base_name = Path(original_filename).stem
            output_filename = f"{base_name}.pdf"
            output_path = self.base_output_dir / category / output_filename
            
            # Handle filename conflicts
            counter = 1
            while output_path.exists():
                output_filename = f"{base_name}_{counter}.pdf"
                output_path = self.base_output_dir / category / output_filename
                counter += 1
            
            # Convert and save as PDF
            FileHandler.images_to_pdf(images, str(output_path))
            logger.info(f"Document saved: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving document: {e}")
            raise

