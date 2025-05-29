from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import io
from utils import predict_on_image, draw_detections, load_model

app = FastAPI(title="LSM-YOLO API", description="API for LSM-YOLO model")

@app.on_event("startup")
async def startup_event():
    try:
        load_model()
    except Exception as e:
        print(f"Error loading model: {e}")


@app.post("/detect_image/",
            tags=["Detection"])
async def detect_roi_and_draw(file: UploadFile = File(...)):
    """
    Nhận file ảnh, thực hiện phát hiện RoI, vẽ bounding box lên ảnh và trả về ảnh đó.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")

    image_bytes = await file.read()
    json_results = predict_on_image(image_bytes)

    if "error" in json_results:
        raise HTTPException(status_code=500, detail=json_results["error"])

    if not json_results.get("detections"):
        # Nếu không có detection, trả về ảnh gốc hoặc thông báo
        # return StreamingResponse(io.BytesIO(image_bytes), media_type=file.content_type)
        return JSONResponse(content={"message": "No detections found.", "detections": []})

    output_image_bytes = draw_detections(image_bytes, json_results["detections"])

    if output_image_bytes:
        return StreamingResponse(io.BytesIO(output_image_bytes), media_type="image/jpeg")
    else:
        raise HTTPException(status_code=500, detail="Failed to draw detections on image.")
    
@app.get("/", tags=["General"])
async def root():
    return {"message": "Welcome to YOLO ROI Detection API! Use /docs for API documentation."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
