import os
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import logging
from io import BytesIO
#import pypdf2
import fitz  # PyMuPDF for better PDF handling

import torch
from transformers import CLIPProcessor, CLIPModel
from PIL import Image, ImageDraw, ImageFont
import pdf2image
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter, A4
import magic
import aiofiles
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentCategories:
    """Define document categories with highly specific prompts for CLIP"""
    
    CATEGORIES = {
    "driving_license": [
        "Driving Licence"
    ],
    "passport": [
      "Passport"
    ],
    "bank_details": [
        "Bank Details", "Passbook", "Account", "Cheque"
    ],
    "employment_verification": [
        "letter on company letterhead stating employee’s job title",
        "Form I-9 section C with employer certification date",
        "work permit showing authorization number and expiry",
        "verification form signed by HR with position start date"
    ],
    "social_security": [
        "card with raised text and official SSA seal",
        "document listing a nine-digit SSN in microprint",
        "laminated card bearing issue date and signature line",
        "SSA card featuring holder’s full legal name in embossed font"
    ],
   
    "tax_documents": [
        "tax",
        "W-2",
        "Form 1040"
    ],
    
    "education_certificate": [
        "Diploma showing institution name, embossed seal, degree awarded, and conferral date",
        "Official transcript with institution letterhead, course–credit table, GPA calculation, and registrar’s signature",
        "Certificate bearing institution seal/stamp, serial number, degree title, and date of conferral",
        "Degree parchment with Board of Education header, authorized signature block, and document ID",
        "Academic marks sheet with institution logo, student details, subject-wise marks table, and registrar sign-off"
    ],
    "others": [
        "Unrelated to driving license, passport, bank details, employment verification, social_security, tax documents, or education certificate",
        "Any random photo if it’s not a document (e.g., pets, landscapes, objects, characters)",
        "Movie, cartoon, or video game character images",
        "Generic correspondence unrelated to official verification",
        "Miscellaneous paper not used for licensing or certification (e.g., receipts, flyers)",
        "Personal note without any government or financial seals",
        "Handwritten memo with no official identifiers",
        "Blank or decorative stationery"
    ]
    }
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
                    # Try with different parameters for better compatibility
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

class CLIPClassifier:
    """CLIP-based document classifier"""
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", cache_dir: str = "./models/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Model cache directory: {self.cache_dir}")
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if model exists locally
        model_path = self.cache_dir / model_name.replace("/", "_")
        
        try:
            if self._model_exists_locally(model_path):
                logger.info(f"Loading cached model from: {model_path}")
                self.model = CLIPModel.from_pretrained(model_path).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_path, use_fast = True)
                logger.info("CLIP model loaded from local cache successfully")
            else:
                logger.info(f"Model not found locally. Downloading {model_name}...")
                self.model = CLIPModel.from_pretrained(model_name, cache_dir=self.cache_dir).to(self.device)
                self.processor = CLIPProcessor.from_pretrained(model_name, cache_dir=self.cache_dir)
                
                # Save model locally for future use
                logger.info(f"Saving model to local cache: {model_path}")
                self.model.save_pretrained(model_path)
                self.processor.save_pretrained(model_path)
                logger.info("CLIP model downloaded and cached successfully")
                
        except Exception as e:
            logger.error(f"Error loading CLIP model: {e}")
            raise
    
    def _model_exists_locally(self, model_path: Path) -> bool:
        """Check if model files exist locally"""
        required_files = [
            "config.json",
            "pytorch_model.bin",
            "preprocessor_config.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json"
        ]
        
        if not model_path.exists():
            return False
            
        for file_name in required_files:
            if not (model_path / file_name).exists():
                logger.info(f"Missing model file: {file_name}")
                return False
                
        logger.info("All required model files found locally")
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
                    probs = logits_per_image.softmax(dim=1)
                    
                image_scores.append(probs.cpu().numpy()[0])
            
            # Average scores across all images
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

# FastAPI Application
app = FastAPI(
    title="Document Classification API",
    description="Classify employee onboarding documents using CLIP",
    version="1.0.0"
)

# Initialize service with model caching
service = DocumentClassificationService(model_cache_dir="./models")

@app.on_event("startup")
async def startup_event():
    logger.info("Document Classification API started")
    logger.info("CLIP model loaded and ready for classification")

@app.post("/classify-document/")
async def classify_document(file: UploadFile = File(...)):
    """
    Classify a single document
    """
    try:
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Process document
            result = await service.process_document(tmp_file_path, file.filename)
            return JSONResponse(content=result)
            
        finally:
            # Clean up temp file
            os.unlink(tmp_file_path)
            
    except Exception as e:
        logger.error(f"Error in classify_document endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify-multiple/")
async def classify_multiple_documents(files: List[UploadFile] = File(...)):
    """
    Classify multiple documents
    """
    results = []
    
    for file in files:
        try:
            if not file.filename:
                results.append({
                    "status": "error",
                    "original_filename": "unknown",
                    "error": "No filename provided"
                })
                continue
            
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                tmp_file_path = tmp_file.name
            
            try:
                # Process document
                result = await service.process_document(tmp_file_path, file.filename)
                results.append(result)
                
            finally:
                # Clean up temp file
                os.unlink(tmp_file_path)
                
        except Exception as e:
            logger.error(f"Error processing file {file.filename}: {e}")
            results.append({
                "status": "error",
                "original_filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse(content={"results": results})

@app.get("/categories/")
async def get_categories():
    """
    Get available document categories
    """
    return JSONResponse(content={
        "categories": list(DocumentCategories.CATEGORIES.keys()),
        "descriptions": DocumentCategories.CATEGORIES
    })



if __name__ == "__main__":
    uvicorn.run(
        "doc_classification:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )