# Troubleshooting Guide

Common issues when setting up vLLM on DGX Spark and their solutions.

## Table of Contents
- [Model Loading Issues](#model-loading-issues)
- [Performance Issues](#performance-issues)
- [Container Issues](#container-issues)
- [HTTPS/Tailscale Issues](#httpstailscale-issues)
- [Memory Issues](#memory-issues)

## Model Loading Issues

### "model type gemma4 not recognized"

**Symptom**:
```
Value error, The checkpoint you are trying to load has model type `gemma4` 
but Transformers does not recognize this architecture.
```

**Cause**: Transformers version too old for Gemma 4 models.

**Solution**:
```bash
# Inside the container
pip install --upgrade transformers
```

Or use a startup script that updates transformers before starting vLLM.

### "Cannot find model"

**Symptom**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/root/.cache/huggingface/...'
```

**Cause**: Volume mount incorrect or model not downloaded.

**Solution**:
```bash
# Ensure cache directory exists on host
mkdir -p ~/.cache/huggingface

# Check the mount path
docker inspect vllm-gemma4-26b | grep -A 5 Mounts
```

### Slow first startup

**Symptom**: First request takes 5-10 minutes.

**Cause**: This is normal! vLLM is:
1. Downloading the model (~15GB)
2. Loading weights (~100s)
3. Compiling CUDA graphs (~55s)

**Solution**: Pre-download the model:
```bash
bash scripts/download-model.sh
```

## Performance Issues

### Very slow inference (~9 tok/s instead of 45+)

**Symptom**: Getting ~9-10 tok/s instead of expected 45+.

**Likely Cause**: Using wrong model or wrong quantization flag.

**Check**:
```bash
curl http://localhost:8000/v1/models | python3 -m json.tool
```

**Fixes**:
1. Ensure you're using `compressed-tensors` quantization:
   ```bash
   --quantization compressed-tensors  # ✅ Fast
   # NOT
   --quantization modelopt            # ❌ Slow on DGX Spark
   ```

2. Ensure CUDA graphs are enabled (check logs for "Capturing CUDA graphs")

3. Verify the model is AEON-7's 26B, not LilaRest's 31B

### "Not enough SMs to use max_autotune_gemm mode"

**Symptom**: Warning in logs.

**Cause**: DGX Spark GB10 has fewer SMs than datacenter GPUs.

**Solution**: This is a warning, not an error. Performance is still optimal.

## Container Issues

### "docker: Error response from daemon: could not select device driver"

**Cause**: NVIDIA Container Toolkit not installed.

**Solution**:
```bash
# Install nvidia-docker2
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

### Container exits immediately

**Symptom**: Container starts then exits with no logs.

**Check**:
```bash
docker logs vllm-gemma4-26b
```

**Common Causes**:
1. Port 8000 already in use
2. Out of disk space
3. Permission denied on cache directory

**Solution**:
```bash
# Check port usage
sudo lsof -i :8000

# Check disk space
df -h

# Fix permissions
sudo chown -R $USER:$USER ~/.cache/huggingface
```

## HTTPS/Tailscale Issues

### "bind() to 100.126.x.x:443 failed"

**Cause**: Port 443 is used by SSH or another service.

**Solution**: Use a different port (e.g., 8443):
```nginx
listen YOUR_TAILSCALE_IP:8443 ssl;
```

### Browser shows "Your connection is not private"

**Cause**: Using self-signed certificate.

**Solution**: This is expected for Tailscale. Click "Advanced" → "Proceed".

Alternatively, get a real certificate:
```bash
sudo tailscale cert your-machine.YOUR_TAILNET.ts.net
```
(Requires Tailscale plan that supports HTTPS certificates)

### Cannot access from other Tailscale machine

**Check**:
1. Both machines show in `tailscale status`
2. No ACL rules blocking port 8000/8443
3. Firewall not blocking on host:
   ```bash
   sudo ufw status
   sudo iptables -L | grep 8000
   ```

## Memory Issues

### "OutOfMemoryError: CUDA out of memory"

**Symptom**:
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Solutions**:

1. Reduce `--gpu-memory-utilization`:
   ```bash
   --gpu-memory-utilization 0.50  # Instead of 0.60
   ```

2. Reduce `--max-model-len`:
   ```bash
   --max-model-len 131072  # Instead of 262000
   ```

3. Reduce `--max-num-seqs`:
   ```bash
   --max-num-seqs 64  # Instead of 128
   ```

### System freezes during startup

**Cause**: Compiling CUDA graphs uses significant CPU/RAM.

**Solution**: This is normal for 30-60 seconds. Wait it out.

If it persists >5 minutes, reduce model length or disable CUDA graphs (will hurt performance):
```bash
--enforce-eager  # Disables CUDA graphs
```

## Getting Help

If issues persist:

1. Check vLLM logs:
   ```bash
   docker logs vllm-gemma4-26b --tail 100
   ```

2. Check GPU status:
   ```bash
   nvidia-smi
   ```

3. Open an issue with:
   - Full error message
   - Output of `docker logs`
   - Output of `nvidia-smi`
   - Your Docker run command
