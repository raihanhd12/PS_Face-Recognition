import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from insightface.app import FaceAnalysis


def compare_faces(img1_path, img2_path):
    # Initialize face analysis model
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))

    # Read images
    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)

    if img1 is None:
        print(f"Error: Cannot load image from {img1_path}")
        return False, 0
    if img2 is None:
        print(f"Error: Cannot load image from {img2_path}")
        return False, 0

    # Detect faces
    faces1 = app.get(img1)
    faces2 = app.get(img2)

    if len(faces1) == 0:
        print(f"Error: No faces detected in {img1_path}")
        return False, 0
    if len(faces2) == 0:
        print(f"Error: No faces detected in {img2_path}")
        return False, 0

    # Get first face from each image
    face1 = faces1[0]
    face2 = faces2[0]

    # Calculate similarity
    sim = np.dot(face1.embedding, face2.embedding) / (
        np.linalg.norm(face1.embedding) * np.linalg.norm(face2.embedding)
    )

    # Threshold for similarity
    threshold = 0.5
    match = sim > threshold

    # Visualize results with bounding boxes
    for face in faces1:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img1, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    for face in faces2:
        bbox = face.bbox.astype(int)
        cv2.rectangle(img2, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

    # Display images with detected faces
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
        f"Similarity: {sim:.4f} ({'MATCH' if match else 'NO MATCH'})", fontsize=16
    )
    plt.tight_layout()
    plt.savefig("comparison_result.jpg")

    print(f"Similarity: {sim:.4f}")
    print(f"Result: {'✅ MATCH' if match else '❌ NO MATCH'}")

    return match, sim


def main():
    # Specify image paths
    img1_path = "image_testing/person2.jpeg"
    img2_path = "image_testing/person7.jpeg"

    # Check if files exist
    if not os.path.exists(img1_path):
        print(f"File not found: {img1_path}")
        return
    if not os.path.exists(img2_path):
        print(f"File not found: {img2_path}")
        return

    # Compare faces
    match, similarity = compare_faces(img1_path, img2_path)


if __name__ == "__main__":
    main()
