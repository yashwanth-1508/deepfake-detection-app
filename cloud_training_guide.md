# Cloud Training Guide: Google Colab & Kaggle

Since training Deep Learning models requires significant GPU power that a standard laptop might lack, using free cloud services is the best solution.

## 1. Using Google Colab (Recommended)
Google Colab provides a free NVIDIA T4 GPU which is perfect for this project.

### Step-by-Step
1.  **Open Colab**: Go to [colab.research.google.com](https://colab.research.google.com).
2.  **Upload the Project**: Create a new notebook and upload your `deepfake-project` folder (zipped) or clone it from GitHub.
3.  **Enable GPU**: Click `Runtime` -> `Change runtime type` -> `T4 GPU`.
4.  **Install Dependencies**:
    ```python
    !pip install torch torchvision opencv-python-headless pillow pytorchcv fastapi uvicorn
    ```
5.  **Run Training**:
    ```python
    # Ensure you are in the project root
    %cd /content/deepfake-project
    !python3 -m model.train kaggle_data/ kaggle_data/metadata.csv
    ```

## 2. Using Kaggle Kernels
Kaggle provides 30 hours of P100 GPU per week.

### Step-by-Step
1.  Go to [Kaggle](https://www.kaggle.com).
2.  Click `Create` -> `New Notebook`.
3.  On the right sidebar, under `Settings`, click `Accelerator` and select `GPU P100`.
4.  Add the `simongraves/deepfake-dataset` to the notebook.
5.  You can use the `prepare_kaggle_data.py` script (modified for Kaggle's internal layout) to set up the CSV and then run training.

## 3. Saving the Results
Once training is finished, download the `model/weights/fine_tuned_head.pth` file and place it in your local `model/weights/` folder for the backend to use.
