import os
import shutil

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.core.security import get_api_key
from app.services.face_recognition import FaceRecognitionService

router = APIRouter()
face_service = FaceRecognitionService()


@router.post("/compare")
async def compare_faces(
    background_tasks: BackgroundTasks,
    image1: UploadFile = File(...),
    image2: UploadFile = File(...),
    threshold: float = 0.5,
    api_key: str = Depends(get_api_key),
):
    """
    Compare two face images and determine if they match

    - **image1**: First face image
    - **image2**: Second face image
    - **threshold**: Similarity threshold for matching (0.0-1.0)
    """
    if not image1.content_type.startswith(
        "image/"
    ) or not image2.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Both files must be images")

    # Create temp directory
    os.makedirs("tmp", exist_ok=True)

    # Save uploaded files
    img1_path = f"tmp/{image1.filename}"
    img2_path = f"tmp/{image2.filename}"

    with open(img1_path, "wb") as buffer:
        shutil.copyfileobj(image1.file, buffer)

    with open(img2_path, "wb") as buffer:
        shutil.copyfileobj(image2.file, buffer)

    # Compare faces
    result, _ = face_service.compare_faces(img1_path, img2_path, threshold=threshold)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    # Schedule cleanup
    def cleanup_files():
        if os.path.exists(img1_path):
            os.remove(img1_path)
        if os.path.exists(img2_path):
            os.remove(img2_path)
        # Visualization will be deleted after serving

    background_tasks.add_task(cleanup_files)

    # Return comparison result with visualization
    if "visualization_path" in result and os.path.exists(result["visualization_path"]):
        return {
            "match": result["match"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
            "visualization": f"/face-recognition/visualization?path={result['visualization_path']}",
        }
    else:
        return {
            "match": result["match"],
            "similarity": result["similarity"],
            "threshold": result["threshold"],
        }


@router.get("/visualization")
async def get_visualization(
    background_tasks: BackgroundTasks, path: str, api_key: str = Depends(get_api_key)
):
    """Get the comparison visualization image"""
    if not os.path.exists(path) or not path.startswith("tmp/"):
        raise HTTPException(status_code=404, detail="Visualization not found")

    # Schedule cleanup after serving
    def cleanup_visualization():
        if os.path.exists(path):
            os.remove(path)

    background_tasks.add_task(cleanup_visualization)

    return FileResponse(path)
