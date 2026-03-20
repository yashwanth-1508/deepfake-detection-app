# Hosting Guide: Deepfake Detection Project

This project contains a **FastAPI backend** (containerized with Docker) and a **Static Frontend**. Here’s how you can host them.

## 1. Backend Deployment (Railway.app - Easiest)

Railway is excellent for Docker-based backend services.

1.  **Create a Railway account** at [railway.app](https://railway.app).
2.  **Install the Railway CLI** (optional but recommended) or connect your GitHub repository.
3.  **New Project** -> **Deploy from GitHub repo**.
4.  Railway will automatically detect the `Dockerfile` and start building the image.
5.  **Expose Port**: Railway usually detects port 8000. If not, set the `PORT` environment variable to `8000` in the Railway dashboard.
6.  **Copy the Public URL**: Once deployed, Railway will provide a URL like `https://deepfake-api.up.railway.app`.

## 2. Frontend Deployment (Vercel)

Vercel is great for hosting static files.

1.  **Create a Vercel account** at [vercel.com](https://vercel.com).
2.  **New Project** -> **Import from GitHub**.
3.  **Select the `frontend` folder** as the root directory of your project.
4.  **Deployment**: Vercel will host your `index.html` and other assets.
5.  **Update API URL**: If you didn't set up a proxy, you might need to update the `API_URL` in `frontend/script.js` to your actual Railway URL.

## 3. Local Testing with Docker

Before deploying, you can test the Docker build locally:

```bash
# Build the image
docker build -t deepfake-api .

# Run the container
docker run -p 8000:8000 deepfake-api
```

Your API will be available at `http://localhost:8000/docs`.

---

### Important Notes
- **Large Models**: The image size might be large (~300MB-500MB) due to PyTorch and OpenCV. Ensure your hosting provider has enough disk space.
- **Inference Speed**: On free-tier CPU hosting, video analysis might take 10-30 seconds. This is normal.
