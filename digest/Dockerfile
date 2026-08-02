FROM python:3.13-slim

# Create the virtual environment in a safe system directory
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the digest folder contents
COPY . .

# Run the script
CMD ["python", "digest.py"]