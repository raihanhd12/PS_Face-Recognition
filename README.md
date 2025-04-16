# Face Recognition API

A FastAPI service for comparing face images and determining if they match using FaceNet PyTorch face recognition technology.

## Features

- Compare two face images to determine if they show the same person
- Returns similarity score and match status
- Generates visualization with detected faces
- Secure API key authentication

## Project Structure

```
.
├── app
│   ├── api
│   │   ├── endpoints
│   │   │   └── face_recognition.py  # API endpoints
│   │   └── api.py                   # Router configuration
│   ├── core
│   │   ├── config.py                # Application settings
│   │   └── security.py              # API key authentication
│   ├── services
│   │   └── face_recognition.py      # Face recognition logic
│   └── main.py                      # Application entry point
├── models                           # Downloaded model files (created automatically)
├── requirements.txt                 # Dependencies
└── README.md                        # This file
```

## Installation

1. Clone the repository
2. Install the dependencies:

```bash
pip install -r requirements.txt
```

The first time you run the application, it will automatically download the required FaceNet models.

## Configuration

1. Generate a secure API key:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

2. Create an `.env` file with your API key:

```
API_KEY=your_generated_key_here
API_KEY_NAME=X-API-Key
API_HOST=0.0.0.0
API_PORT=8000
```

## Usage

1. Start the server:

```bash
python main.py
```

2. Access the API documentation at http://localhost:8000/docs
3. Make API requests with your API key:

```bash
# Using curl
curl -X POST "http://localhost:8000/face-recognition/compare" \
  -H "X-API-Key: your_api_key_here" \
  -F "image1=@person1.jpg" \
  -F "image2=@person2.jpg"
```

```python
# Using Python requests
import requests

headers = {"X-API-Key": "your_api_key_here"}
files = {
    "image1": open("person1.jpg", "rb"),
    "image2": open("person2.jpg", "rb")
}

response = requests.post(
    "http://localhost:8000/face-recognition/compare",
    headers=headers,
    files=files
)
print(response.json())
```

## API Endpoints

### POST /face-recognition/compare

Compare two face images to determine if they match.

**Parameters:**

- `image1`: First face image (file upload)
- `image2`: Second face image (file upload)
- `threshold`: Similarity threshold for matching (0.0-1.0, default: 0.7)

**Headers:**

- `X-API-Key`: Your API key

**Response:**

```json
{
  "match": true,
  "similarity": 0.9245,
  "threshold": 0.7,
  "visualization": "/face-recognition/visualization?path=tmp/comparison_abc123.jpg"
}
```

### GET /face-recognition/visualization

Get the visualization image showing the comparison.

**Parameters:**

- `path`: Path to visualization image (returned from compare endpoint)

**Headers:**

- `X-API-Key`: Your API key

**Response:**

- Image file (JPEG)

## Dependencies

- FaceNet PyTorch: Face recognition models (MTCNN and InceptionResnetV1)
- FastAPI: Web framework
- PyTorch: Deep learning framework
- OpenCV: Image processing
- Matplotlib: Visualization generation