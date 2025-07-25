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
    allow_origins=["https://localhost:3000", "https://127.0.0.1:4000"],
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
    use_textline_orientation=False,
    # use_gpu=True
)

# ocr = paddleocr.PPStructureV3(
#     use_doc_orientation_classify=False,
#     use_doc_unwarping=False
# )


@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    print(f"--- New request received for file: {file.filename} ---")
    if not file.filename.endswith(".pdf"):
        print("Error: File is not a PDF.")
        return JSONResponse(status_code=400, content={"message": "Please upload a PDF file."})

    try:
        print("Step 1: Reading PDF content...")
        pdf_contents = await file.read()
        print("Step 2: Converting PDF to images...")
        images = convert_from_bytes(pdf_contents, dpi=300)
        print(f"Step 3: PDF converted to {len(images)} image(s).")

        all_page_results = []
        for i, image in enumerate(images):
            print(f"\n--- Processing page {i + 1} ---")
            image_path = f"temp_{uuid.uuid4()}.png"
            print(f"Step 4: Saving page {i + 1} to temporary image: {image_path}")
            image.save(image_path, "PNG")

            print(f"Step 5: Running OCR on page {i + 1}...")
            results_for_page = ocr.predict(input=image_path)
            print(f"Step 6: OCR found {len(results_for_page)} text blocks on page {i + 1}.")

            page_content = []
            for j, res in enumerate(results_for_page):
                print(f"  - Processing text block {j + 1} on page {i + 1}...")
                with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as temp_json:
                    res.save_to_json(temp_json.name)
                    temp_json.seek(0)
                    text_block_data = json.load(temp_json)
                page_content.append(text_block_data)
                os.remove(temp_json.name)

            all_page_results.append({f"page_{i+1}": page_content})
            print(f"Step 7: Cleaning up temporary image file for page {i + 1}.")
            os.remove(image_path)

        print("\nStep 8: All pages processed. Returning final JSON response.")
        return JSONResponse(status_code=200, content={"results": all_page_results})

    except Exception as e:
        print(f"An error occurred: {e}")
        return JSONResponse(status_code=500, content={"message": f"An error occurred during processing: {e}"})

    # return JSONResponse(status_code=200, content={"results": all_results})


from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
