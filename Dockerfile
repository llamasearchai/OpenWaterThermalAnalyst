# OpenWaterThermalAnalyst Dockerfile with OrbStack optimization
FROM python:3.9-slim

# Set environment variables for better Python performance
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=random \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies for geospatial libraries
RUN apt-get update && apt-get install -y \
    build-essential \
    gdal-bin \
    libgdal-dev \
    libspatialindex-dev \
    libgeos-dev \
    libproj-dev \
    proj-data \
    proj-bin \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set GDAL environment variables
ENV GDAL_VERSION=3.6.4 \
    GDAL_CONFIG=/usr/bin/gdal-config \
    GEOS_CONFIG=/usr/bin/geos-config

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash openwater

# Set working directory
WORKDIR /app

# Copy dependency files first for better caching
COPY pyproject.toml ./

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install -e .[ai,dev,viz,hydro]

# Copy source code
COPY src/ ./src/
COPY config/ ./config/
COPY tests/ ./tests/

# Create necessary directories
RUN mkdir -p /app/data /app/output /app/logs && \
    chown -R openwater:openwater /app

# Switch to non-root user
USER openwater

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command runs the FastAPI server
CMD ["python", "-m", "open_water_thermal_analyst.api"]
