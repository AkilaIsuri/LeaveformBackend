from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import asyncio
from azure.core.credentials import AzureKeyCredential
from azure.ai.formrecognizer import DocumentAnalysisClient
import os
from dotenv import load_dotenv
import base64
import io
from concurrent.futures import ThreadPoolExecutor
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('invoice_processor.log')
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate environment variables
required_env_vars = ["AZURE_ENDPOINT", "AZURE_KEY"]
missing_vars = [var for var in required_env_vars if not os.getenv(var)]
if missing_vars:
    raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

app = FastAPI()

# Initialize ThreadPoolExecutor for CPU-bound tasks
executor = ThreadPoolExecutor(max_workers=4)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Azure Configuration
try:
    document_analysis_client = DocumentAnalysisClient(
        endpoint=os.getenv("AZURE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("AZURE_KEY"))
    )
    logger.info("Successfully initialized Document Analysis Client")
except Exception as e:
    logger.error(f"Failed to initialize Document Analysis Client: {str(e)}")
    raise

async def extract_first_page(file_content: bytes) -> str:
    """Extract first page from PDF and return as a thumbnail using PyMuPDF."""
    def _extract():
        try:
            logger.info(f"Starting PDF first page extraction with content size: {len(file_content)} bytes")
            
            # Import fitz (PyMuPDF) inside the function to handle any import issues gracefully
            import fitz
            
            # Load PDF from bytes
            pdf = fitz.open(stream=file_content, filetype="pdf")
            
            if len(pdf) == 0:
                logger.error("PDF has no pages")
                return None
            
            # Get first page
            page = pdf[0]
            logger.info(f"Processing page {page.number + 1}, size: {page.rect}")
            
            # Render page to an image with higher resolution for better quality
            # Increase the matrix values for higher resolution
            zoom_factor = 2.0  # Adjust as needed for quality vs file size
            mat = fitz.Matrix(zoom_factor, zoom_factor)
            pix = page.get_pixmap(matrix=mat, alpha=False)
            
            logger.info(f"Generated pixmap with dimensions: {pix.width}x{pix.height}")
            
            # Convert to PNG data
            img_bytes = pix.tobytes("png")
            
            # Convert to base64
            thumbnail = base64.b64encode(img_bytes).decode('utf-8')
            logger.info(f"Generated thumbnail of size: {len(thumbnail)} bytes")
            
            return thumbnail
                
        except ImportError as e:
            logger.error(f"PyMuPDF import error: {str(e)}. Please install with 'pip install pymupdf'")
            return None
        except Exception as e:
            logger.error(f"PDF extraction error: {str(e)}", exc_info=True)
            # Try to log more details about the file
            logger.error(f"PDF content size: {len(file_content)} bytes")
            if len(file_content) > 0:
                # Log the first few bytes of the file to help diagnose issues
                try:
                    logger.error(f"PDF content first 100 bytes (hex): {file_content[:100].hex()}")
                except:
                    pass
            return None

    return await asyncio.get_event_loop().run_in_executor(executor, _extract)

async def process_invoice(file_content: bytes, filename: str) -> dict:
    """Process invoice with Azure Document Intelligence."""
    try:
        logger.info(f"Starting invoice analysis for file: {filename}")
        
        poller = document_analysis_client.begin_analyze_document(
            "prebuilt-invoice",  # Using prebuilt-invoice model
            document=file_content
        )
        result = poller.result()
        
        # Field mapping as requested
        field_mapping = {
            'InvoiceId': 'PV Number',
            'InvoiceDate': 'Date Prepared',
            'RemittanceAddressRecipient': 'Supplier or Company'
        }
        
        # Initialize response structure
        document_data = {}
        
        # Add the default Document Type field
        document_data['Document Type'] = {
            "value": "Payable Voucher",
            "confidence": 1.0  # Default value has full confidence
        }
        
        # Initialize default values for required fields
        document_data['PV Number'] = {"value": None, "confidence": 0.0}
        document_data['Supplier or Company'] = {"value": None, "confidence": 0.0}
        document_data['Date Prepared'] = {"value": None, "confidence": 0.0}
        
        # Process document if available
        if hasattr(result, 'documents') and result.documents:
            for document in result.documents:
                fields = document.fields
                
                # Log the fields found by Azure
                logger.info(f"Fields found by Azure: {[key for key in fields]}")
                
                # Process each field with the mapping
                for api_field, display_field in field_mapping.items():
                    if api_field in fields:
                        field = fields[api_field]
                        
                        # Extract value based on field type
                        if hasattr(field, 'value_string') and field.value_string is not None:
                            value = field.value_string
                        elif hasattr(field, 'value_date') and field.value_date is not None:
                            value = str(field.value_date)
                        elif hasattr(field, 'content') and field.content is not None:
                            value = field.content
                        else:
                            value = None
                            
                        document_data[display_field] = {
                            "value": value,
                            "confidence": field.confidence
                        }
                        logger.info(f"Extracted {display_field}: {value} with confidence {field.confidence}")
        
        logger.info(f"Extracted document data: {document_data}")
        return document_data

    except Exception as e:
        logger.error(f"Invoice processing error for {filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/process-document")
async def process_document(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    token: str = Form(...)
):
    """Process a single invoice document with thumbnail generation."""
    try:
        logger.info(f"Processing invoice: {file.filename}")
        
        file_content = await file.read()
        metadata_dict = json.loads(metadata)
        
        # Process invoice and generate thumbnail concurrently
        extracted_data_task = process_invoice(file_content, file.filename)
        thumbnail_task = extract_first_page(file_content)
        
        # Wait for both tasks to complete
        extracted_data, thumbnail = await asyncio.gather(
            extracted_data_task,
            thumbnail_task
        )
        
        if thumbnail is None:
            logger.warning(f"Thumbnail generation failed for {file.filename}")
        
        response_data = {
            "filename": file.filename,
            "extracted_data": extracted_data,
            "thumbnail": thumbnail,
            "metadata": metadata_dict
        }
        
        return JSONResponse(
            content=response_data,
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "POST, OPTIONS",
                "Access-Control-Allow-Headers": "*"
            }
        )
    
    except Exception as e:
        logger.error(f"Error processing document {file.filename}: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI application")
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")