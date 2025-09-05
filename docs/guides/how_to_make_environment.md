# How to make environment

This article explains how to set up TensorFlow environment using Docker in Windows system (or Linux, because we use Docker).

> Disclaimer: The content of this article may be incorrect :/

> Used Prompt: `docker를 이용해 tensorflow image를 사용하는 방법에 대해 적어 줘. 완전 차근차근 적어야 해.`

## Method 1: Using Docker Compose (Recommended)

Docker Compose provides the most convenient way to manage TensorFlow containers, especially for development environments.

### Prerequisites

- Windows 10/11 or Linux system
- Administrative privileges to install software
- Stable internet connection for downloading Docker images
- NVIDIA GPU and drivers (for GPU support)

### Step 1: Install Docker

#### For Windows

1. Go to <https://docs.docker.com/engine/install/>
2. Download Docker Desktop for Windows
3. Run the installer as administrator
4. Follow the installation wizard
5. Restart your computer when prompted
6. Start Docker Desktop from the Start menu

#### For Linux

Follow the installation guide for your specific distribution at <https://docs.docker.com/engine/install/>

### Step 2: Verify Docker Installation

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

### Step 3: Create Directory Structure

Create the necessary directories for your project:

```bash
mkdir notebooks data models src
```

For Windows PowerShell:

```powershell
New-Item -ItemType Directory -Path notebooks, data, models, src
```

### Step 4: Run TensorFlow Services

#### Option A: Run GPU-enabled TensorFlow with Jupyter (Default/Recommended)

```bash
docker-compose --profile gpu up tensorflow-gpu
```

#### Option B: Run CPU-only TensorFlow with Jupyter

```bash
docker-compose up tensorflow-jupyter
```

#### Option C: Run CPU-only TensorFlow for script execution

```bash
docker-compose --profile cpu-only up tensorflow-cpu
```

#### Option D: Run in background (detached mode)

```bash
docker-compose --profile gpu up -d tensorflow-gpu
```

### Step 5: Access Jupyter Notebook

- **Jupyter Notebook (GPU)**: <http://localhost:8889>
- **Jupyter Notebook (CPU)**: <http://localhost:8888>

To find the token for Jupyter, check the container logs:

```bash
docker-compose logs tensorflow-gpu
```

You'll see output like:

```txt
To access the notebook, open this file in a browser:
    file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
Or copy and paste one of these URLs:
    http://127.0.0.1:8889/?token=abc123...
```

Copy the URL with the token and paste it into your web browser.

### Step 6: Verify TensorFlow Installation

Create a new notebook and test TensorFlow:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Simple test
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy().decode('utf-8'))
```

### Step 7: Manage Services

#### Stop services

```bash
docker-compose down
```

#### Stop and remove volumes

```bash
docker-compose down -v
```

#### View running services

```bash
docker-compose ps
```

#### View logs

```bash
docker-compose logs -f tensorflow-gpu
```

#### Restart services

```bash
docker-compose restart tensorflow-gpu
```

### Step 8: Execute Commands in Running Container

#### Open interactive Python shell

```bash
docker-compose exec tensorflow-gpu python
```

#### Run a specific Python script

```bash
docker-compose exec tensorflow-gpu python /tf/notebooks/your_script.py
```

#### Open bash shell in container

```bash
docker-compose exec tensorflow-gpu bash
```

### Advantages of Docker Compose Method

1. **Configuration Management**: All settings are stored in a single file
2. **Easy Service Management**: Start/stop multiple services with simple commands
3. **Environment Consistency**: Reproducible development environment
4. **Volume Management**: Automatic directory creation and mounting
5. **Network Management**: Automatic network creation between services
6. **Profile Support**: Switch between CPU and GPU configurations easily

### Useful Docker Compose Commands

```bash
# Build and start services
docker-compose --profile gpu up --build

# Scale services (run multiple instances)
docker-compose --profile gpu up --scale tensorflow-gpu=2

# Pull latest images
docker-compose pull

# Show service configuration
docker-compose config

# Remove stopped containers
docker-compose rm

# View resource usage
docker-compose top
```

### Example Workflow

1. Start Docker Desktop
2. Navigate to your project directory in PowerShell
3. Create directories:

   ```powershell
   New-Item -ItemType Directory -Path notebooks, data, models, src
   ```

4. Start GPU Jupyter service:

   ```bash
   docker-compose --profile gpu up tensorflow-gpu
   ```

5. Open the provided URL in your browser
6. Start coding with TensorFlow!

### Troubleshooting

#### Docker Compose not found

- Make sure Docker Desktop is installed and running
- Docker Compose is included with Docker Desktop

#### Permission denied errors

- Run PowerShell as administrator
- Check Docker Desktop settings for file sharing permissions

#### Container exits immediately

- Check if Docker Desktop is running
- Verify the docker-compose.yml syntax
- Use `docker-compose logs <service_name>` to see error messages

#### Port already in use

- Change the port mapping in docker-compose.yml:

  ```yaml
  ports:
    - "8890:8888"  # Use different host port
  ```

#### GPU not detected

- Install NVIDIA Docker runtime
- Verify NVIDIA drivers are installed
- Check if GPU is available: `nvidia-smi`

This Docker Compose method provides a complete, easily manageable TensorFlow environment that's perfect for development and experimentation.
      - ./models:/tf/models
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - JUPYTER_ENABLE_LAB=yes
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    profiles:
      - gpu
    command: >
      bash -c "
        mkdir -p /tf/notebooks /tf/data /tf/models &&
        jupyter notebook --notebook-dir=/tf/notebooks --ip=0.0.0.0 --no-browser --allow-root
      "

```

### Step 4: Create Directory Structure

Create the necessary directories for your project:

```bash
mkdir notebooks data models src
```

For Windows PowerShell:

```powershell
New-Item -ItemType Directory -Path notebooks, data, models, src
```

### Step 5: Run TensorFlow Services

#### Option A: Run Jupyter with CPU-only TensorFlow (Default)

```bash
docker-compose up tensorflow-jupyter
```

#### Option B: Run CPU-only TensorFlow for script execution

```bash
docker-compose --profile cpu-only up tensorflow-cpu
```

#### Option C: Run GPU-enabled TensorFlow (requires NVIDIA GPU)

```bash
docker-compose --profile gpu up tensorflow-gpu
```

#### Option D: Run in background (detached mode)

```bash
docker-compose up -d tensorflow-jupyter
```

### Step 6: Access Jupyter Notebook

- **Jupyter Notebook (CPU)**: <http://localhost:8888>
- **Jupyter Notebook (GPU)**: <http://localhost:8889>

To find the token for Jupyter, check the container logs:

```bash
docker-compose logs tensorflow-jupyter
```

You'll see output like:

```txt
To access the notebook, open this file in a browser:
    file:///root/.local/share/jupyter/runtime/nbserver-1-open.html
Or copy and paste one of these URLs:
    http://127.0.0.1:8888/?token=abc123...
```

Copy the URL with the token and paste it into your web browser.

### Step 7: Verify TensorFlow Installation

Create a new notebook and test TensorFlow:

```python
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Simple test
hello = tf.constant('Hello, TensorFlow!')
print(hello.numpy().decode('utf-8'))
```

### Step 8: Manage Services

#### Stop services

```bash
docker-compose down
```

#### Stop and remove volumes

```bash
docker-compose down -v
```

#### View running services

```bash
docker-compose ps
```

#### View logs

```bash
docker-compose logs -f tensorflow-jupyter
```

#### Restart services

```bash
docker-compose restart tensorflow-jupyter
```

### Step 9: Execute Commands in Running Container

#### Open interactive Python shell

```bash
docker-compose exec tensorflow-jupyter python
```

#### Run a specific Python script

```bash
docker-compose exec tensorflow-jupyter python /tf/notebooks/your_script.py
```

#### Open bash shell in container

```bash
docker-compose exec tensorflow-jupyter bash
```

### Advantages of Docker Compose Method

1. **Configuration Management**: All settings are stored in a single file
2. **Easy Service Management**: Start/stop multiple services with simple commands
3. **Environment Consistency**: Reproducible development environment
4. **Volume Management**: Automatic directory creation and mounting
5. **Network Management**: Automatic network creation between services
6. **Profile Support**: Switch between CPU and GPU configurations easily

### Useful Docker Compose Commands

```bash
# Build and start services
docker-compose up --build

# Scale services (run multiple instances)
docker-compose up --scale tensorflow-jupyter=2

# Pull latest images
docker-compose pull

# Show service configuration
docker-compose config

# Remove stopped containers
docker-compose rm

# View resource usage
docker-compose top
```

### Example Workflow

1. Start Docker Desktop
2. Navigate to your project directory in PowerShell
3. Create docker-compose.yml file (copy from above)
4. Create directories:

   ```powershell
   New-Item -ItemType Directory -Path notebooks, data, models, src
   ```

5. Start Jupyter service:

   ```bash
   docker-compose up tensorflow-jupyter
   ```

6. Open the provided URL in your browser
7. Start coding with TensorFlow!

### Troubleshooting

#### Docker Compose not found

- Make sure Docker Desktop is installed and running
- Docker Compose is included with Docker Desktop

#### Permission denied errors

- Run PowerShell as administrator
- Check Docker Desktop settings for file sharing permissions

#### Container exits immediately

- Check if Docker Desktop is running
- Verify the docker-compose.yml syntax
- Use `docker-compose logs <service_name>` to see error messages

#### Port already in use

- Change the port mapping in docker-compose.yml:

```yaml
ports:
  - "8890:8888"  # Use different host port
```

#### GPU not detected

- Install NVIDIA Docker runtime
- Verify NVIDIA drivers are installed
- Check if GPU is available: `nvidia-smi`

This Docker Compose method provides a complete, easily manageable TensorFlow environment that's perfect for development and experimentation.

## Method 2: Using Docker Run Commands (Alternative)

If you prefer using Docker run commands directly, you can also use the traditional approach. However, we recommend the Docker Compose method above for better management and consistency.

For quick reference on Docker run commands, you can use:

```bash
# Run TensorFlow with GPU and Jupyter
docker run -it --rm --gpus all -p 8888:8888 -v ${PWD}:/tf/notebooks tensorflow/tensorflow:latest-gpu-jupyter

# Run TensorFlow CPU only
docker run -it --rm tensorflow/tensorflow:latest python

# Run with custom script
docker run -it --rm -v ${PWD}:/tf/workdir -w /tf/workdir tensorflow/tensorflow:latest python your_script.py
```

However, for production and development environments, the Docker Compose approach is strongly recommended.
