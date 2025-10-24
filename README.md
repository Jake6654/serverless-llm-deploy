# 🧠 Serverless LLM Deployment Guide (VM Environment)

This repository documents the full process of deploying **ServerlessLLM** on a virtual machine (VM) with GPU support.  
It includes driver installation, Docker Compose setup, environment variable configuration, and model deployment.

---

## 📋 Environment Overview

- **Base OS:** Debian / Ubuntu 22.04 (Deep Learning VM)
- **GPU:** NVIDIA L4 (or equivalent)
- **CUDA Version:** 12.1+
- **Docker:** Installed with Compose plug-in
- **Python:** 3.11+
- **ServerlessLLM Image:** `serverlessllm/sllm:latest`

---

## 1️⃣ Install NVIDIA Driver + DKMS Module

```bash
sudo /opt/deeplearning/install-driver.sh
sudo reboot
nvidia-smi
```

✅ **Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.1     |
| GPU Name        Persistence-M    Bus-Id    Disp.A    Volatile Uncorr. ECC   |
| 0 NVIDIA L4     On               00000000:00:1E.0    Off                    |
+-----------------------------------------------------------------------------+
```

If you see something like above, the GPU driver is properly installed.

---

## 2️⃣ Install Docker Compose Plug-in

```bash
sudo apt-get update && sudo apt-get install -y docker-compose-plugin
docker compose version
```

Sometimes, this can cause a **backports** error.  
You can resolve it with the following:

```bash
# 1. Check for "bullseye-backports" references
grep -R "bullseye-backports" -n /etc/apt/sources.list /etc/apt/sources.list.d || true

# 2. Comment out backports lines
sudo sed -i.bak '/bullseye-backports/s/^/#/' /etc/apt/sources.list 2>/dev/null || true
sudo sed -i.bak '/bullseye-backports/s/^/#/' /etc/apt/sources.list.d/*.list 2>/dev/null || true

# 3. Update sources
sudo apt-get update
```

Then reinstall:

```bash
sudo apt-get install -y docker-compose-plugin
```

---

## 3️⃣ Check GPU Runtime (NVIDIA Toolkit required)

```bash
docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi
```

✅ **Expected output:**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.3     |
| GPU Name        Persistence-M    Bus-Id    Disp.A    Volatile Uncorr. ECC   |
| 0 NVIDIA L4     On               00000000:00:1E.0    Off                    |
+-----------------------------------------------------------------------------+
```

If you see `NVIDIA-SMI` output, the GPU runtime is available inside Docker.

---

## 4️⃣ Project Directory & Environment Variables

```bash
mkdir -p ~/serverless-llm/models
cd ~/serverless-llm

cat > .env << 'EOF'
MODEL_FOLDER=/home/jaehyukc1223/serverless-llm/models
EOF

# Fix potential permission issues
sudo chown -R $USER:$USER ~/serverless-llm
chmod -R u+rwx ~/serverless-llm
```

---

## 5️⃣ Docker Compose Configuration

Create a `docker-compose.yml` in the project root:

```yaml
services:
  sllm_head:
    image: serverlessllm/sllm:latest
    container_name: sllm_head
    shm_size: '16gb'       # Increased from 8GB for Ray stability
    mem_limit: 14g
    environment:
      - MODEL_FOLDER=${MODEL_FOLDER}
      - MODE=HEAD
    ports:
      - "6379:6379"
      - "8343:8343"
    networks:
      - sllm_network
    restart: unless-stopped

  sllm_worker_0:
    image: serverlessllm/sllm:latest
    container_name: sllm_worker_0
    shm_size: '16gb'
    mem_limit: 32g
    runtime: nvidia
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              capabilities: ["gpu"]
              device_ids: ["0"]
    environment:
      - WORKER_ID=0
      - STORAGE_PATH=/models
      - MODE=WORKER
    networks:
      - sllm_network
    volumes:
      - ${MODEL_FOLDER}:/models
    command: ["--mem-pool-size","2GB","--registration-required"]
    restart: unless-stopped

networks:
  sllm_network:
    driver: bridge
    name: sllm
```

---

## 6️⃣ Start the Cluster

```bash
docker compose up -d
docker compose ps
```

✅ **Expected output:**
```
NAME             COMMAND                  STATE     PORTS
sllm_head        "python /app/serve.py"   Up        6379/tcp, 8343/tcp
sllm_worker_0    "python /app/serve.py"   Up
```

Wait for all images to download and check if containers are in **Started** state.

---

## 7️⃣ Activate Python Virtual Environment

```bash
python3 -m venv ~/.venvs/sllm
source ~/.venvs/sllm/bin/activate
pip install --upgrade pip
pip install serverless-llm
```

---

## 8️⃣ Deploy Model

```bash
sllm-cli deploy --model facebook/opt-125m
```

✅ If successful, you’ll see deployment logs showing `facebook/opt-125m` being loaded.

---

## 9️⃣ Test Chat Completion API

```bash
curl -X POST "http://127.0.0.1:8343/v1/chat/completions" -H "Content-Type: application/json" -d '{
  "model": "facebook/opt-1.3b",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is your name?"}
  ],
  "temperature": 0.7,
  "max_tokens": 100
}'
```

✅ **Expected JSON response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1739999999,
  "model": "facebook/opt-1.3b",
  "choices": [
    {
      "message": {"role": "assistant", "content": "I’m Serverless LLM, nice to meet you!"},
      "finish_reason": "stop",
      "index": 0
    }
  ]
}
```

---

## 🧩 Notes

- Tested on **Google Cloud VM (NVIDIA L4, CUDA 12.3)**
- Docker image: `serverlessllm/sllm:latest`
- If `nvidia-smi` doesn’t work inside the container, reinstall the NVIDIA container toolkit.
- You can modify the model path or scale worker nodes by editing the `docker-compose.yml`.

---

## 📘 Credits

Guide prepared by **Jae-Hyuk Chang**  
for the **ServerlessLLM Research Project (ShenGroup, University of Virginia)**  
_Last updated: October 2025_
