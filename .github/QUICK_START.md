# 🚀 Quick Start Guide

## Immediate Usage

Your GitHub Actions are ready to use! Here's how to get started:

### 1. Push to Trigger Build
```bash
git add .
git commit -m "Add GitHub Actions for multi-arch Docker builds"
git push origin main
```

### 2. Monitor Build Progress
- Go to your repository on GitHub
- Click the **Actions** tab
- Watch your first multi-arch build run!

### 3. Access Your Images
After successful build, your images will be available at:
```
ghcr.io/${{ github.repository }}:latest
```

Example: `ghcr.io/mister2d/coexistai:latest`

### 4. Test Locally
```bash
# Pull and run your image
docker run -p 8000:8000 ghcr.io/mister2d/coexistai:latest

# Test the endpoints
curl http://localhost:8000/
curl http://localhost:8000/health
```

## 🎯 What You Get

✅ **Multi-architecture support**: AMD64, ARM64
✅ **Automatic builds**: On every push to main/master  
✅ **Security scanning**: Vulnerability detection with Trivy  
✅ **SBOM generation**: Software Bill of Materials  
✅ **Intelligent tagging**: Latest, version tags, PR tags  
✅ **Free hosting**: GitHub Container Registry  
✅ **Public runners**: No infrastructure needed  

## 📋 Available Workflows

1. **Main Build** (`docker-build-multiarch.yml`)
   - Full multi-arch build and push
   - Security scanning and testing
   - Runs on: push, PR, tags, manual

2. **Local Test** (`docker-test-local.yml`)
   - Quick AMD64-only build
   - Local testing without push
   - Runs on: PR affecting Docker files

3. **Release** (`release.yml`)
   - Semantic version management
   - GitHub release creation
   - Runs on: version tags

## 🔧 Customization

### Change Platforms
Edit `.github/workflows/docker-build-multiarch.yml`:
```yaml
platforms: linux/amd64,linux/arm64  # Remove arm/v7 if needed
```

### Add Custom Tags
Modify the metadata action in workflows:
```yaml
tags: |
  type=raw,value=custom-tag
  type=ref,event=branch
```

### Environment Variables
Add to your repository secrets:
- `DOCKERHUB_USERNAME` - For Docker Hub sync
- `DOCKERHUB_PASSWORD` - For Docker Hub sync

## 🚨 Troubleshooting

### Build Fails?
1. Check the Actions tab for error logs
2. Verify your Dockerfile builds locally:
   ```bash
   docker build -f config/Dockerfile -t test .
   ```

### Permission Issues?
- Go to Settings → Actions → General
- Enable "Read and write permissions"

### Image Not Found?
- Wait for build to complete (5-15 minutes)
- Check package settings: Settings → Packages
- Ensure visibility is set correctly

## 📚 Next Steps

1. **Read the full documentation**: See [`docs/GITHUB_ACTIONS.md`](docs/GITHUB_ACTIONS.md)
2. **Customize workflows**: Edit files in `.github/workflows/`
3. **Set up notifications**: Configure GitHub notifications
4. **Monitor usage**: Check Actions usage in repository settings

## 🎉 Success!

Your CoexistAI application is now set up with professional-grade CI/CD that:
- Builds for multiple architectures automatically
- Hosts images in GitHub Container Registry
- Runs security scans on every build
- Tests images before deployment
- Manages releases with semantic versioning

**No local infrastructure required** - everything runs on GitHub's free public runners!

Happy building! 🚀