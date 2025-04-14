import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from insightface.app import FaceAnalysis

class FaceRecognitionService:
    def __init__(self):
        # Initialize face analysis model once
        self.app = FaceAnalysis(providers=["CPUExecutionProvider"])
        self.app.prepare(ctx_id=-1, det_size=(640, 640))

    def compare_faces(self, img1_path, img2_path, threshold=0.5):
        """Compare two face images and return match result and similarity score"""
        # Read images
        img1 = cv2.imread(img1_path)
        img2 = cv2.imread(img2_path)

        if img1 is None or img2 is None:
            return {"error": "Cannot load one or both images"}, 0

        # Detect faces
        faces1 = self.app.get(img1)
        faces2 = self.app.get(img2)

        if len(faces1) == 0 or len(faces2) == 0:
            return {"error": "No faces detected in one or both images"}, 0

        # Get first face from each image
        face1 = faces1[0]
        face2 = faces2[0]

        # Calculate similarity
        similarity = np.dot(face1.embedding, face2.embedding) / (
            np.linalg.norm(face1.embedding) * np.linalg.norm(face2.embedding)
        )

        # Determine match
        match = similarity > threshold

        # Create result visualization
        result_img_path = self._create_comparison_image(img1, img2, faces1, faces2, similarity, match)

        return {
            "match": bool(match),
            "similarity": float(similarity),
            "threshold": threshold,
            "visualization_path": result_img_path
        }, similarity

    def _create_comparison_image(self, img1, img2, faces1, faces2, similarity, match):
        """Create a visualization image showing the comparison"""
        # Draw bounding boxes
        for face in faces1:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        for face in faces2:
            bbox = face.bbox.astype(int)
            cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

        # Create comparison visualization
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        plt.title("Image 1")
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        plt.title("Image 2")
        plt.axis("off")

        plt.suptitle(
            f"Similarity: {similarity:.4f} ({'MATCH' if match else 'NO MATCH'})", fontsize=16
        )
        plt.tight_layout()

        # Ensure directory exists
        os.makedirs("tmp", exist_ok=True)
        result_path = f"tmp/comparison_{os.urandom(4).hex()}.jpg"
        plt.savefig(result_path)
        plt.close()

        return result_path