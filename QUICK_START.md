# SmartDiet ML API - Quick Start Guide

## 🚀 Fast Setup (3 commands)

```bash
# 1. Copy environment template
cp .env.example .env

# 2. Build images (FIRST TIME: 10-30 min, AFTER: 30 sec!)
DOCKER_BUILDKIT=1 docker-compose build

# 3. Start services
docker-compose up -d
```

**Access:**
- 📚 API Docs: http://localhost:8000/docs
- 🌐 Web Interface: http://localhost:7860

## ⚡ Quick Commands (Use Makefile)

```bash
make build      # Build with cache (FAST!)
make up         # Start services
make down       # Stop services
make logs       # View logs
make rebuild    # Force full rebuild (slow)
make clean      # Remove everything
```

## 🔧 Enable BuildKit (One-Time Setup)

**Required for fast caching!**

```bash
# Option 1: Add to ~/.bashrc or ~/.zshrc (recommended)
echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
source ~/.bashrc

# Option 2: Or set in .env file
echo "DOCKER_BUILDKIT=1" >> .env
```

## 📦 What Was Fixed

### 1. Docker Caching (90-95% faster rebuilds!)
- ✅ BuildKit cache mounts for pip
- ✅ Optimized layer caching
- ✅ Persistent cache across builds
- 📖 See: `DOCKER_CACHING_GUIDE.md`

### 2. Import Paths
- ✅ Fixed circular imports
- ✅ Corrected module paths
- ✅ All imports now work in Docker

### 3. Docker Configuration
- ✅ Fixed port conflicts
- ✅ Proper build contexts
- ✅ Volume mounts for data/cache
- ✅ Service dependencies

### 4. Configuration Files
- ✅ Docker-compatible paths
- ✅ .env.example template
- ✅ .dockerignore for both services

📖 **Full details:** `FIXES_SUMMARY.md`

## 🐛 Troubleshooting

### Cache not working?
```bash
# Check BuildKit is enabled
docker version | grep BuildKit

# Should see: "BuildKit: moby/buildkit"
```

### Still downloading packages?
```bash
# First build will download everything (one time)
# Second build should use cache and be MUCH faster
# If not, check: DOCKER_BUILDKIT=1 is set
```

### Port already in use?
```bash
# Change ports in .env
echo "ML_API_PORT=8001" >> .env
echo "WEB_PORT=7861" >> .env
```

## 📊 Performance

**Before optimization:**
```
docker-compose build
# Time: 10-30 minutes (every build)
```

**After optimization:**
```
# First build
DOCKER_BUILDKIT=1 docker-compose build
# Time: 10-30 minutes (one time)

# Subsequent builds
DOCKER_BUILDKIT=1 docker-compose build
# Time: 30 seconds - 2 minutes 🚀
```

## 🔍 Project Structure

```
smartdiet_ml_api/
├── app/                # ML API Service (FastAPI + FAISS)
│   ├── Dockerfile      # With BuildKit cache
│   └── requirements.txt # 1384 lines (cached!)
├── web/                # Gradio Web Interface
│   ├── Dockerfile      # With BuildKit cache
│   └── requirements.txt
├── docker-compose.yaml # Orchestration config
├── .env.example        # Environment template
├── Makefile           # Quick commands
└── DOCKER_CACHING_GUIDE.md # Deep dive
```

## 📚 Documentation

- `DOCKER_CACHING_GUIDE.md` - How caching works + troubleshooting
- `FIXES_SUMMARY.md` - All fixes applied to the project
- `CLAUDE.md` - Project overview for AI assistants

## 💡 Tips

1. **Enable BuildKit globally** (saves typing)
   ```bash
   echo 'export DOCKER_BUILDKIT=1' >> ~/.bashrc
   ```

2. **Use Makefile shortcuts**
   ```bash
   make build  # Instead of: DOCKER_BUILDKIT=1 docker-compose build
   make up     # Instead of: docker-compose up -d
   ```

3. **Check cache usage**
   ```bash
   make cache-info
   ```

4. **Clean cache if needed**
   ```bash
   make clean-cache
   ```

## 🎯 Next Steps

1. ✅ Setup complete? Test it:
   ```bash
   curl http://localhost:8000/health
   ```

2. 📸 Upload image via web: http://localhost:7860

3. 🔧 Modify code? Rebuild is now FAST:
   ```bash
   make build && make restart
   ```

---

**Questions?** Check `DOCKER_CACHING_GUIDE.md` or `FIXES_SUMMARY.md`
