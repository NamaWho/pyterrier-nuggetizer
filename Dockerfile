FROM pytorch/pytorch

# Install dependencies
RUN apt-get update && apt-get install --no-install-recommends -y \
    build-essential \
    curl \
    wget \
    unzip \
    nano \
    git \
    whiptail \
    software-properties-common \
    nvtop \
    libncurses5 \
    && rm -rf /var/lib/apt/lists/* 

# Install optional utilities
RUN pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

CMD ["/bin/bash"]