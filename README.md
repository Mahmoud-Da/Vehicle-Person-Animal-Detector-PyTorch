# Vehicle, Person, and Animal Detector with PyTorch

![GitHub top language](https://img.shields.io/github/languages/top/your-username/your-repo-name?style=flat-square)
![GitHub last commit](https://img.shields.io/github/last-commit/your-username/your-repo-name?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=flat-square)

This project uses a pre-trained Faster R-CNN model from PyTorch's `torchvision` library to detect and classify objects in images. It identifies a wide range of objects from the COCO dataset and groups them into three main categories: **vehicles**, **people**, and **animals**, drawing colored bounding boxes around each.

The project supports two main workflows:

1.  **Direct Inference:** A fast and easy method that uses the original pre-trained model directly. (Recommended)
2.  **Fine-Tuning:** An advanced method to re-train the model on a specific subset of the data for potentially higher accuracy.

## Demo

![Demo Image](![street_detected_20250610_004600](https://github.com/user-attachments/assets/3033a12e-f865-41fa-99e1-2a717a47e4c4)
)
_An example output image with bounding boxes. **Vehicles** are green, **people** are red, and **animals** are yellow._

## Features

- **Multi-Class Detection:** Detects objects and classifies them into `vehicle`, `person`, or `animal`.
- **High-Performance Model:** Leverages the powerful Faster R-CNN with a ResNet50 backbone, pre-trained on the COCO dataset.
- **Configurable:** Easily change confidence thresholds, colors, and default images via the `config.py` file.
- **Two Usage Modes:** Choose between quick direct inference or advanced fine-tuning.
- **Saves Output:** Automatically saves the resulting images with detections to the `outputs` directory.

## Project Structure

```
.
├── config.py               # Main configuration file for paths, colors, etc.
├── detect.py               # Main script for DIRECT INFERENCE (quick use).
├── helpers.py              # Helper functions (e.g., saving images).
├── inputs/                 # Directory to place your images for detection.
│   └── street.jpg
└── outputs/                # Directory where detected images are saved.

# --- Optional Files for Fine-Tuning (Advanced) ---
├── train.py                # Script to fine-tune the model.
├── detect_finetuned.py     # Script to run inference with your fine-tuned model.
├── dataset.py              # PyTorch dataset class for COCO.
├── download_data.py        # Script to download the COCO dataset (~20 GB).
├── engine.py               # Helper scripts for training & evaluation.
└── utils.py                #
```

## Setup and Installation

**1. Clone the Repository**

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

**2. Create a Virtual Environment**
It's highly recommended to use a virtual environment.

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**3. Install Dependencies**
Install all the required Python libraries from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

_(If a `requirements.txt` file is not provided, create one with the following content or install manually)_

```txt
# requirements.txt
torch
torchvision
opencv-python
numpy
pycocotools
```

## Usage

This project offers two distinct methods for object detection.

---

### Approach 1: Direct Inference (Recommended for Quick Use)

This method uses the original, highly-capable COCO pre-trained model without any re-training. It's fast, requires no data download, and is very effective for common scenarios.

**1. Place Images in the `inputs` Folder**
Add any `.jpg` or `.png` images you want to process into the `inputs/` directory.

**2. Run the Detection Script**
Execute `detect.py` from your terminal. By default, it will process the image specified in `config.py`.

```bash
python3 detect.py
```

_(Note: The first time you run this, it will download the pre-trained model weights (~160 MB). This only happens once.)_

**3. Check the `outputs` Folder**
The resulting image, with bounding boxes and labels, will be saved in the `outputs/` directory with `_output` appended to its name.

---

### Approach 2: Fine-Tuning the Model (Advanced)

This method involves re-training (fine-tuning) the model's final layers on a specific subset of the COCO dataset. This can potentially improve accuracy for your target classes but requires significant time and disk space.

**⚠️ Warning:** This process requires downloading **~20 GB** of data and can take several hours to train, depending on your hardware.

**1. Download the COCO 2017 Dataset**
Run the provided helper script to download and extract the training and validation data.

```bash
python3 download_data.py
```

**2. Download Training Helper Scripts**
The training script depends on standard helper files from the official PyTorch repository. A download script may be provided, or you may need to fetch them manually from [here](https://github.com/pytorch/vision/tree/main/references/detection).

**3. Start the Training Process**
Once the data and helpers are in place, run the training script.

```bash
python3 train.py
```

This will train the model for a number of epochs and save the fine-tuned weights as a `.pth` file (e.g., `vehicle_person_animal_detector.pth`).

**4. Run Inference with the Fine-Tuned Model**
Use a separate detection script (e.g., `detect_finetuned.py`) that is specifically designed to load your custom-trained `.pth` file and perform inference.

## Configuration

You can easily customize the detector's behavior by editing the `config.py` file:

- `CONFIDENCE_THRESHOLD`: Set the minimum score for a detection to be displayed (e.g., `0.6` means 60% confidence).
- `DEFAULT_IMAGE_NAME`: Change the default image to be processed from the `inputs` folder.
- `CATEGORY_TO_COLOR`: Change the bounding box colors for each category. Note that OpenCV uses **BGR** (Blue, Green, Red) format.
  ```python
  CATEGORY_TO_COLOR = {
      'vehicle': (0, 255, 0),    # Green
      'person':  (0, 0, 255),    # Red
      'animal':  (23, 232, 255)  # A nice yellow
  }
  ```

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

# Running Your PyTorch Project with Docker

This document outlines the steps to build and run this PyTorch application using Docker and Docker Compose. This ensures a consistent and reproducible environment for development and deployment.

## Prerequisites

1.  **Docker**: Ensure Docker Desktop (for Mac/Windows) or Docker Engine (for Linux) is installed and running. You can download it from [docker.com](https://www.docker.com/products/docker-desktop/).
2.  **Docker Compose**: Docker Compose V2 is typically included with Docker Desktop. For Linux, you might need to install it separately.
3.  **(Optional) NVIDIA GPU Support**:
    - If you intend to use NVIDIA GPUs, ensure you have the latest NVIDIA drivers installed on your host machine.
    - Install the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) on your host machine. This allows Docker containers to access NVIDIA GPUs.
4.  **Project Files**:
    - `Dockerfile`: Defines the Docker image for the application.
    - `docker-compose.yml`: Defines how to run the application services (including GPU support).
    - `Pipfile`: Specifies Python package dependencies.
    - `Pipfile.lock`: Locks package versions for reproducible builds.
    - Your application code (e.g., `inference.py`).

## Building and Running the Application

We will use Docker Compose to manage the build and run process.

### Step 1: Clone the Repository (if applicable)

If you haven't already, clone the project repository to your local machine:

```bash
git clone <your-repository-url>
cd <your-project-directory>
```

### Step 2: Check/Generate Pipfile.lock

The `Dockerfile` uses `pipenv install --deploy`, which requires `Pipfile.lock` to be up-to-date with `Pipfile`.

**Troubleshooting `Pipfile.lock` out-of-date error:**
If, during the Docker build process (Step 3), you encounter an error similar to:

```
Your Pipfile.lock (...) is out of date. Expected: (...).
ERROR:: Aborting deploy
```

This means your `Pipfile.lock` is not synchronized with your `Pipfile`. To fix this, run the following command in your project's root directory (where `Pipfile` is located) on your **host machine**:

```bash
pipenv lock
```

This will update `Pipfile.lock`. After running this command, proceed to Step 3.

### Step 3: Build and Run with Docker Compose

Open your terminal in the root directory of the project (where `docker-compose.yml` and `Dockerfile` are located).

**To build the image and run the application (e.g., execute `inference.py`):**

```bash
docker-compose up --build
```

- `--build`: This flag tells Docker Compose to build the Docker image using the `Dockerfile`. You can omit this on subsequent runs if the `Dockerfile` or its dependencies haven't changed, and an image already exists.
- The application (defined by `CMD` in the `Dockerfile`, e.g., `python3 inference.py`) will start, and its output will be displayed in your terminal.

**To run in detached mode (in the background):**

```bash
docker-compose up --build -d
```

### Step 4: Interacting with the Application

- **Viewing Logs (if running in detached mode):**

  ```bash
  docker-compose logs -f app
  ```

  (Replace `app` with your service name if it's different in `docker-compose.yml`). Press `Ctrl+C` to stop following logs.

- **Accessing a Shell Inside the Container (for debugging):**
  If you need to explore the container's environment or run commands manually:

  1.  Ensure the container is running (e.g., using `docker-compose up -d`).
  2.  Open a shell:
      ```bash
      docker-compose exec app bash
      ```
      (Replace `app` with your service name if it's different).
  3.  Inside the container, you can navigate to `/app` (the working directory) and run Python scripts or other commands.

- **Port Mapping (if applicable):**
  If your application (`inference.py`) runs a web server (e.g., on port 8000) and you have configured port mapping in `docker-compose.yml` (e.g., `ports: - "8000:8000"`), you can access it via `http://localhost:8000` in your web browser.

### Step 5: Stopping the Application

To stop and remove the containers, networks, and (optionally, depending on `docker-compose down` flags) volumes defined by Docker Compose:

```bash
docker-compose down
```

If you want to remove the volumes as well:

```bash
docker-compose down -v
```

## Important Notes

- **PyTorch Versions & CUDA:** The `Pipfile` specifies PyTorch versions and a CUDA source (`pytorch-cu111`). Ensure these versions are valid and available from the specified PyTorch wheel index. If `pipenv install` fails during the Docker build due to version conflicts or "Could not find a version" errors, you will need to:
  1.  Consult [PyTorch Previous Versions](https://pytorch.org/get-started/previous-versions/) to find compatible `torch`, `torchvision`, and `torchaudio` versions for your desired CUDA version (e.g., CUDA 11.1).
  2.  Update the versions in your `Pipfile`.
  3.  Run `pipenv lock` locally to regenerate `Pipfile.lock`.
  4.  Re-run `docker-compose up --build`.
- **GPU Usage:** The `docker-compose.yml` is configured to attempt GPU access using NVIDIA. This requires the prerequisites mentioned above (NVIDIA drivers and NVIDIA Container Toolkit on the host). If GPUs are not available or not configured correctly, PyTorch will typically fall back to CPU mode.
- **Development Mode Volume Mount:** The `docker-compose.yml` includes `volumes: - .:/app`. This mounts your local project directory into the container. Code changes made locally will be reflected inside the container, which is useful for development. For production, you might remove this volume mount to rely solely on the code baked into the image.

## Further Actions

- **Cleaning up Docker Resources:**
  - To remove unused Docker images: `docker image prune`
  - To remove unused Docker volumes: `docker volume prune`
  - To remove unused Docker networks: `docker network prune`
  - To remove all unused Docker resources (images, containers, volumes, networks): `docker system prune -a` (Use with caution!)
