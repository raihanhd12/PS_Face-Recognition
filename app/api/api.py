from fastapi import APIRouter
from app.api.endpoints import face_recognition

api_router = APIRouter()
api_router.include_router(face_recognition.router, prefix="/face-recognition", tags=["Face Recognition"])