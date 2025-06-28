# Docker Compose Usage Guide

This guide explains how to use the various Docker Compose profiles and services for STAMP benchmarking.

## 🐳 Available Profiles

### Benchmark Profiles
- **`benchmark`**: Run all frameworks with mixed precision
- **`pytorch`**: Run only PyTorch benchmarks
- **`tensorflow`**: Run only TensorFlow benchmarks  
- **`jax`**: Run only JAX benchmarks
- **`interactive`**: Interactive shell for custom benchmark runs

### Analysis Profiles
- **`dashboard`**: Streamlit dashboard for results visualization
- **`jupyter`**: Jupyter Lab for interactive analysis
- **`full`**: Dashboard + Jupyter Lab together

## 🚀 Usage Examples

### Quick Start - Run All Benchmarks
```bash
# Run comprehensive benchmarks across all frameworks
docker-compose --profile benchmark up benchmark-runner

# Run and remove containers after completion
docker-compose --profile benchmark up --rm benchmark-runner
```

### Framework-Specific Benchmarks
```bash
# Run only PyTorch benchmarks
docker-compose --profile pytorch up pytorch-benchmark

# Run only TensorFlow benchmarks  
docker-compose --profile tensorflow up tensorflow-benchmark

# Run only JAX benchmarks
docker-compose --profile jax up jax-benchmark
```

### Interactive Analysis
```bash
# Start Streamlit dashboard
docker-compose --profile dashboard up streamlit-dashboard
# Access at: http://localhost:8501

# Start Jupyter Lab
docker-compose --profile jupyter up jupyter-lab
# Access at: http://localhost:8888

# Start both dashboard and Jupyter
docker-compose --profile full up
```

### Custom Benchmark Runs
```bash
# Interactive shell for custom configurations
docker-compose --profile interactive up benchmark-interactive

# Inside the container, run custom benchmarks:
python3 main.py --framework pytorch --precision fp16 --verbose
python3 main.py --framework all --precision mixed --output-dir custom_logs
```

## 🔧 Advanced Usage

### Build and Run with Custom Arguments
```bash
# Build fresh images and run
docker-compose --profile benchmark build
docker-compose --profile benchmark up --build benchmark-runner

# Run specific service with custom environment
docker-compose run --rm \
  -e NVIDIA_VISIBLE_DEVICES=0 \
  benchmark-runner python3 main.py --framework pytorch --precision fp32
```

### Volume Management
```bash
# View created volumes
docker volume ls

# Remove all volumes (clears logs and data)
docker-compose down --volumes

# Backup logs
docker run --rm -v stamp-benchmark_logs:/data -v $(pwd):/backup \
  ubuntu tar czf /backup/logs_backup.tar.gz -C /data .
```

### Multi-Service Orchestration
```bash
# Run benchmarks and start dashboard simultaneously
docker-compose --profile benchmark --profile dashboard up

# Run everything except interactive services
docker-compose --profile full --profile benchmark up
```

## 📊 Monitoring and Debugging

### Container Logs
```bash
# Follow logs from benchmark runner
docker-compose --profile benchmark logs -f benchmark-runner

# View logs from all services
docker-compose logs

# Check specific service status
docker-compose ps
```

### Resource Monitoring
```bash
# Monitor resource usage
docker stats

# Check GPU usage inside container
docker-compose --profile interactive exec benchmark-interactive nvidia-smi
```

### Debugging Failed Runs
```bash
# Run interactive shell for debugging
docker-compose --profile interactive run --rm benchmark-interactive bash

# Check system info inside container
docker-compose --profile interactive run --rm benchmark-interactive \
  python3 -c "from utils.profiler import SystemProfiler; print(SystemProfiler.get_quick_stats())"
```

## 🎯 Production Recommendations

### For CI/CD Pipelines
```bash
# Automated benchmark runs
docker-compose --profile benchmark up --abort-on-container-exit benchmark-runner

# Exit with container exit code
docker-compose --profile benchmark run --rm benchmark-runner
```

### For Development
```bash
# Keep services running for development
docker-compose --profile full --profile interactive up -d

# Access interactive shell while other services run
docker-compose exec benchmark-interactive bash
```

### For Batch Processing
```bash
# Run multiple configurations sequentially
docker-compose --profile pytorch up --rm pytorch-benchmark
docker-compose --profile tensorflow up --rm tensorflow-benchmark
docker-compose --profile jax up --rm jax-benchmark
```

## 🛠️ Troubleshooting

### Common Issues

**GPU not accessible:**
```bash
# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi

# Verify NVIDIA Container Toolkit installation
sudo docker run --rm --gpus all nvidia/cuda:12.2-base-ubuntu22.04 nvidia-smi
```

**Port conflicts:**
```bash
# Use different ports
docker-compose run --rm -p 8502:8501 streamlit-dashboard
```

**Permission issues:**
```bash
# Fix volume permissions
sudo chown -R $USER:$USER logs/ data/
```

**Memory issues:**
```bash
# Limit container memory
docker-compose run --rm --memory=8g benchmark-runner
```

## 🔄 Cleanup

```bash
# Stop all services
docker-compose down

# Remove containers, networks, and volumes
docker-compose down --volumes --remove-orphans

# Clean up Docker system
docker system prune -a
```
