# How to make environment

This article explains how to set up TensorFlow environment using Docker in Windows system (or Linux, because we use Docker).

> Disclaimer: The content of this article may be incorrect :/

> Used Prompt: `docker를 이용해 tensorflow image를 사용하는 방법에 대해 적어 줘. 완전 차근차근 적어야 해.`

## Prerequisites

- Windows 10/11 or Linux system
- Administrative privileges to install software
- Stable internet connection for downloading Docker images

## Step 1: Install Docker

### For Windows

1. Go to <https://docs.docker.com/engine/install/>
2. Download Docker Desktop for Windows
3. Run the installer as administrator
4. Follow the installation wizard
5. Restart your computer when prompted
6. Start Docker Desktop from the Start menu

### For Linux

Follow the installation guide for your specific distribution at <https://docs.docker.com/engine/install/>

## Step 2: Verify Docker Installation

Open PowerShell (Windows) or Terminal (Linux) and run:

```bash
docker --version
```

You should see output similar to:

```txt
Docker version 24.0.6, build ed223bc
```

Also verify Docker is running:

```bash
docker run hello-world
```

## Step 3: Pull TensorFlow Docker Image

There are several TensorFlow Docker images available. Choose based on your needs:

### Option A: TensorFlow with Jupyter (Recommended for development)

```bash
docker pull tensorflow/tensorflow:latest-jupyter
```

### Option B: TensorFlow CPU only

```bash
docker pull tensorflow/tensorflow:latest
```

### Option C: TensorFlow with GPU support (requires NVIDIA GPU and drivers)

```bash
docker pull tensorflow/tensorflow:latest-gpu-jupyter
```

## Step 4: Create and Run TensorFlow Container

### Option A: Run with Jupyter Notebook (Recommended)

Create a container with Jupyter notebook access:

```bash
docker run -it --rm -p 8888:8888 -v ${PWD}:/tf/notebooks tensorflow/tensorflow:latest-jupyter
```

For Windows PowerShell, use:

```powershell
docker run -it --rm -p 8888:8888 -v "${PWD}:/tf/notebooks" tensorflow/tensorflow:latest-jupyter
```

### Option B: Run Interactive Python Shell

```bash
docker run -it --rm tensorflow/tensorflow:latest python
```

### Option C: Run with Custom Script

If you have a Python script to run:

```bash
docker run -it --rm -v ${PWD}:/tf/workdir -w /tf/workdir tensorflow/tensorflow:latest python your_script.py
```

For Windows PowerShell:

```powershell
docker run -it --rm -v "${PWD}:/tf/workdir" -w /tf/workdir tensorflow/tensorflow:latest python your_script.py
```

## Step 5: Access Jupyter Notebook (If using Option A)

1. After running the Jupyter container, you'll see output like:

   ```txt
   To access the notebook, open this file in a browser:
       file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
   Or copy and paste one of these URLs:
       http://127.0.0.1:8888/?token=abc123...
   ```

2. Copy the URL with the token and paste it into your web browser
3. You can now create and run Jupyter notebooks with TensorFlow

## Step 6: Verify TensorFlow Installation

Create a new notebook or Python script and test TensorFlow:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Simple test
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy().decode('utf-8'))
```

## Common Docker Commands for TensorFlow

### List running containers

```bash
docker ps
```

### List all containers (including stopped)

```bash
docker ps -a
```

### Stop a running container

```bash
docker stop <container_id>
```

### Remove a container

```bash
docker rm <container_id>
```

### List downloaded images

```bash
docker images
```

### Remove an image

```bash
docker rmi tensorflow/tensorflow:latest
```

## Important Notes

1. **Data Persistence**: Use volume mounting (`-v` flag) to persist your work outside the container
2. **Port Mapping**: Use `-p` flag to access services like Jupyter from your host machine
3. **GPU Support**: For GPU support, you need NVIDIA Docker runtime and compatible GPU drivers
4. **Memory Usage**: TensorFlow containers can use significant memory, ensure you have adequate RAM

## Troubleshooting

### Docker Desktop not starting

- Make sure virtualization is enabled in BIOS
- Run Docker Desktop as administrator
- Check Windows features: Hyper-V and WSL 2

### Permission denied errors

- Run PowerShell as administrator
- Check Docker Desktop settings for file sharing permissions

### Container exits immediately

- Check if Docker Desktop is running
- Verify the image name is correct
- Use `docker logs <container_id>` to see error messages

## Example Workflow

1. Start Docker Desktop
2. Pull TensorFlow image:

   ```bash
   docker pull tensorflow/tensorflow:latest-jupyter
   ```

3. Navigate to your project directory in PowerShell
4. Run container with Jupyter:

   ```powershell
   docker run -it --rm -p 8888:8888 -v "${PWD}:/tf/notebooks" tensorflow/tensorflow:latest-jupyter
   ```

5. Open the provided URL in your browser
6. Start coding with TensorFlow!

This setup provides a complete, isolated TensorFlow environment that's consistent across different systems.
