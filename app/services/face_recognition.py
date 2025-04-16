import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image


class FaceRecognitionService:
    def __init__(self):
        # Use GPU if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Initialize face detector
        self.detector = MTCNN(
            image_size=160,
            margin=0,
            min_face_size=20,
            thresholds=[0.6, 0.7, 0.7],
            factor=0.709,
            device=self.device,
        )

        # Initialize face recognition model (pretrained on VGGFace2)
        self.model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)

    def compare_faces(self, img1_path, img2_path, threshold=0.7):
        """Compare two face images and return match result and similarity score"""
        try:
            # Get face embeddings
            embedding1, face_box1 = self.get_face_embedding(img1_path)
            embedding2, face_box2 = self.get_face_embedding(img2_path)

            # Calculate cosine similarity (convert to similarity score)
            cos_similarity = np.dot(embedding1, embedding2) / (
                np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
            )
            similarity = (cos_similarity + 1) / 2  # Convert from [-1,1] to [0,1]

            # Determine if faces match
            match = similarity > threshold

            # Read images for visualization
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)

            # Convert box format for visualization
            # MTCNN returns [x1, y1, x2, y2]
            # We need [x1, y1, x2, y2] for rectangle drawing
            face1_box = [
                int(face_box1[0]),
                int(face_box1[1]),
                int(face_box1[2]),
                int(face_box1[3]),
            ]
            face2_box = [
                int(face_box2[0]),
                int(face_box2[1]),
                int(face_box2[2]),
                int(face_box2[3]),
            ]

            # Create visualization
            result_img_path = self._create_comparison_image(
                img1, img2, face1_box, face2_box, similarity, match
            )

            return {
                "match": bool(match),
                "similarity": float(similarity),
                "threshold": threshold,
                "visualization_path": result_img_path,
            }, similarity

        except Exception as e:
            return {"error": f"Error during face comparison: {str(e)}"}, 0

    def get_face_embedding(self, image_path):
        """Extract face embedding and bounding box from image"""
        # Check if image exists
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        # Load image
        img = Image.open(image_path)

        # Detect face and get bounding boxes
        boxes, _ = self.detector.detect(img)

        if boxes is None or len(boxes) == 0:
            raise ValueError(f"No face detected in {image_path}")

        # Use the first face
        box = boxes[0]

        # Get aligned face
        face = self.detector(img)

        if face is None:
            raise ValueError(f"Failed to align face in {image_path}")

        # Get embedding
        with torch.no_grad():
            embedding = self.model(face.unsqueeze(0).to(self.device))

        return embedding[0].cpu().numpy(), box

    def _create_comparison_image(
        self, img1, img2, face1_box, face2_box, similarity, match
    ):
        """Create a visualization image showing the comparison"""
        # Draw bounding boxes
        cv2.rectangle(
            img1,
            (face1_box[0], face1_box[1]),
            (face1_box[2], face1_box[3]),
            (0, 255, 0),
            2,
        )

        cv2.rectangle(
            img2,
            (face2_box[0], face2_box[1]),
            (face2_box[2], face2_box[3]),
            (0, 255, 0),
            2,
        )

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
            f"Similarity: {similarity:.4f} ({'MATCH' if match else 'NO MATCH'})",
            fontsize=16,
        )
        plt.tight_layout()

        # Ensure directory exists
        os.makedirs("tmp", exist_ok=True)
        result_path = f"tmp/comparison_{os.urandom(4).hex()}.jpg"
        plt.savefig(result_path)
        plt.close()

        return result_path
