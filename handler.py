import logging
from typing import List
from PIL import Image
from io import BytesIO
import magic
import fitz 
import pdf2image
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileHandler:
    """Handle file format conversion and processing"""
    
    SUPPORTED_FORMATS = {
        'image': ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'],
        'pdf': ['.pdf'],
        'document': ['.doc', '.docx', '.txt']
    }
    
    @staticmethod
    def get_file_type(file_path: str) -> str:
        """Determine file type using python-magic"""
        try:
            mime_type = magic.from_file(file_path, mime=True)
            return mime_type
        except Exception as e:
            logger.error(f"Error detecting file type: {e}")
            # Fallback to extension
            ext = Path(file_path).suffix.lower()
            if ext in FileHandler.SUPPORTED_FORMATS['image']:
                return 'image'
            elif ext in FileHandler.SUPPORTED_FORMATS['pdf']:
                return 'application/pdf'
            return 'unknown'
    
    @staticmethod
    def convert_to_images(file_path: str) -> List[Image.Image]:
        """Convert various file formats to PIL Images"""
        images = []
        
        try:
            # First, try to determine if it's a PDF by extension (more reliable)
            file_extension = Path(file_path).suffix.lower()
            logger.info(f"Processing file: {file_path} with extension: {file_extension}")
            
            if file_extension == '.pdf':
                # Handle PDF files - convert directly to images
                logger.info(f"Converting PDF to images: {file_path}")
                try:
                
                    pdf_images = pdf2image.convert_from_path(
                        file_path,
                        dpi=200,  # Reduced DPI for better compatibility
                        first_page=1,
                        last_page=None,
                        fmt='JPEG',
                        thread_count=1,
                        userpw=None,
                        use_cropbox=False,
                        strict=False
                    )
                    
                    # Convert each page to RGB
                    for i, pdf_img in enumerate(pdf_images):
                        try:
                            rgb_img = pdf_img.convert('RGB')
                            images.append(rgb_img)
                            logger.info(f"Converted PDF page {i+1} to image")
                        except Exception as page_error:
                            logger.error(f"Error converting PDF page {i+1}: {page_error}")
                            continue
                    
                    if images:
                        logger.info(f"Successfully converted PDF to {len(images)} image(s)")
                    else:
                        raise ValueError("No pages could be converted from PDF")
                        
                except Exception as pdf_error:
                    logger.error(f"Error converting PDF to images: {pdf_error}")
                    # Try alternative method with PyMuPDF
                    try:
                        logger.info("Trying alternative PDF conversion with PyMuPDF...")
                        pdf_document = fitz.open(file_path)
                        for page_num in range(pdf_document.page_count):
                            page = pdf_document[page_num]
                            # Render page as image
                            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom
                            pix = page.get_pixmap(matrix=mat)
                            img_data = pix.tobytes("ppm")
                            img = Image.open(BytesIO(img_data)).convert('RGB')
                            images.append(img)
                            logger.info(f"Converted PDF page {page_num+1} using PyMuPDF")
                        pdf_document.close()
                        
                        if images:
                            logger.info(f"Successfully converted PDF using PyMuPDF: {len(images)} image(s)")
                        else:
                            raise ValueError("PyMuPDF conversion failed - no images extracted")
                            
                    except Exception as pymupdf_error:
                        logger.error(f"PyMuPDF conversion also failed: {pymupdf_error}")
                        raise ValueError(f"Failed to convert PDF with both methods: pdf2image({pdf_error}) and PyMuPDF({pymupdf_error})")
                
            elif file_extension in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']:
                # Handle image files
                logger.info(f"Processing image file: {file_path}")
                try:
                    img = Image.open(file_path).convert('RGB')
                    images.append(img)
                    logger.info("Successfully loaded image file")
                except Exception as img_error:
                    logger.error(f"Error loading image: {img_error}")
                    raise ValueError(f"Failed to load image file: {img_error}")
                
            else:
                logger.error(f"Unsupported file extension: {file_extension}")
                raise ValueError(f"Unsupported file type. Supported formats: PDF, JPG, JPEG, PNG, BMP, TIFF, GIF")
                
        except Exception as e:
            logger.error(f"Error in convert_to_images: {e}")
            if "Failed to convert PDF" in str(e) or "PyMuPDF conversion" in str(e):
                raise e  # Re-raise specific PDF errors
            else:
                raise ValueError(f"Could not process document: {e}")
            
        if not images:
            raise ValueError("No images were extracted from the document")
            
        logger.info(f"Final result: {len(images)} images extracted")
        return images
    
    @staticmethod
    def images_to_pdf(images: List[Image.Image], output_path: str):
        """Convert list of PIL Images to PDF"""
        try:
            if len(images) == 1:
                images[0].save(output_path, "PDF", resolution=100.0)
            else:
                images[0].save(
                    output_path, "PDF", 
                    resolution=100.0, 
                    save_all=True,
                    append_images=images[1:]
                )
        except Exception as e:
            logger.error(f"Error converting images to PDF: {e}")
            raise