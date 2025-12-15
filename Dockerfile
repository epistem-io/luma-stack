# Use Python 3.9 slim image
FROM python:3.9-slim

# Install minimal system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
RUN useradd -m -u 1000 user

# Set working directory
WORKDIR /home/user/app

# Copy requirements and install Python dependencies
COPY --chown=user:user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire application
COPY --chown=user:user . .

# Install the epistemx package in development mode
RUN pip install -e .

# Get Google Service Account JSON secret and OAuth2 client config at buildtime
RUN --mount=type=secret,id=GOOGLE_SERVICE_ACCOUNT_JSON_B64,mode=0444,required=false \
    --mount=type=secret,id=GOOGLE_OAUTH_CLIENT_CONFIG_B64,mode=0444,required=false \
    mkdir -p /home/user/app/auth && \
    if [ -f /run/secrets/GOOGLE_SERVICE_ACCOUNT_JSON_B64 ]; then \
        base64 -d /run/secrets/GOOGLE_SERVICE_ACCOUNT_JSON_B64 > /home/user/app/auth/service-account.json; \
    fi && \
    if [ -f /run/secrets/GOOGLE_OAUTH_CLIENT_CONFIG_B64 ]; then \
        base64 -d /run/secrets/GOOGLE_OAUTH_CLIENT_CONFIG_B64 > /home/user/app/auth/oauth_client_config.json; \
    fi && \
    chown -R user:user /home/user/app/auth

# Create temp directory for leafmap and ensure proper permissions
RUN mkdir -p /home/user/tmp && \
    chown -R user:user /home/user/app && \
    chown -R user:user /home/user/tmp

# Switch to non-root user
USER user

# Set environment variables
ENV HOME=/home/user
ENV PATH=/home/user/.local/bin:$PATH
ENV TMPDIR=/home/user/tmp
ENV TEMP=/home/user/tmp

# Expose the port that Streamlit runs on
EXPOSE 7860

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Health check
HEALTHCHECK CMD curl --fail http://localhost:7860/_stcore/health

# Command to run the application
CMD ["streamlit", "run", "home.py", "--server.port", "7860", "--server.address", "0.0.0.0", "--server.enableXsrfProtection", "false"]