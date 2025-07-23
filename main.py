from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import paddleocr
from pdf2image import convert_from_bytes
import os
import uuid
import json
import tempfile

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Server is working!"}

ocr = paddleocr.PaddleOCR(
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False
)

@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"PDF received: {file.filename}")
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"message": "Please upload a PDF file."})

    try:
        pdf_contents = await file.read()
        images = convert_from_bytes(pdf_contents)

        all_page_results = []
        for i, image in enumerate(images):
            image_path = f"temp_{uuid.uuid4()}.png"
            image.save(image_path, "PNG")

            # Perform OCR. `predict` returns a list of result objects for the page.
            results_for_page = ocr.predict(input=image_path)

            page_content = []
            # Each `res` is a result object for a detected text block
            for res in results_for_page:
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_json:
                    res.save_to_json(temp_json.name)
                    temp_json.seek(0)
                    text_block_data = json.load(temp_json)
                page_content.append(text_block_data)
                os.remove(temp_json.name) # Clean up the temp json for this block

            all_page_results.append({f"page_{i+1}": page_content})

            # Clean up the temporary image file
            os.remove(image_path)

        return JSONResponse(status_code=200, content={"results": all_page_results})

    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred during processing: {e}"})

    # return JSONResponse(status_code=200, content={"results": all_results})

