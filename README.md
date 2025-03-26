**OpenNuggetizer: An Open-Source Framework for Evaluating Retrieval-Augmented Generation (RAG) Systems**

OpenNuggetizer is an open-source tool designed to assess the factual accuracy of Retrieval-Augmented Generation (RAG) systems by generating, scoring, and assigning information nuggets to RAG outputs. Inspired by the original [Nuggetizer framework](https://arxiv.org/pdf/2411.09607), OpenNuggetizer exclusively utilizes open-source inference models, ensuring accessibility and adaptability for the research community.

**Key Features:**

- **Information Nugget Generation:** Automatically extract concise information units ("nuggets") from textual data to facilitate detailed evaluation.

- **Nugget Scoring and Assignment:** Assess and align these nuggets with RAG-generated responses to measure factual consistency and relevance.

- **Open-Source Integration:** Leverage open-source Large Language Models (LLMs) and Natural Language Processing (NLP) tools to ensure transparency and reproducibility.

---

**Getting Started:**

This guide explains how to set up and run OpenNuggetizer using Docker in a secure and interactive container environment.

### 1. Clone the Repository

```bash
git clone <REPO_URL>
cd <REPO_NAME>
```

### 2. Build the Docker Image

Build the Docker container with the required dependencies:

```bash
docker build -t nuggetizer .
```

### 3. Run the Docker Container

Run the container in interactive mode with GPU support, mounting the `data` directory:

```bash
docker run -it --gpus all --name nuggetizer -v "$(pwd)/data:/app/data" -d nuggetizer /bin/bash
```

Alternatively, if you want to specify a particular GPU device:

```bash
docker run -it --gpus "device=0" --name nuggetizer -v "$(pwd)/data:/app/data" -d nuggetizer /bin/bash
```

### 4. Access the Running Container

Open an interactive shell inside the container:

```bash
docker exec -it nuggetizer bash
```

### 5. Start the Model Server

Within the container, start the `vllm` server with your model and API key:

```bash
vllm serve <MODEL_NAME> --dtype auto --api-key=<API_KEY> --port 8080
```

### 6. Generate Nuggets

After the model server is running, execute the nugget generation script:

```bash
PYTHONPATH=. python3 scripts/create_nuggets.py \
--input_file data/retrieval_results/<INPUT_FILE> \
--output_file data/nuggets/<OUTPUT_FILE> \
--log_level 2
```

---

**Contributing:**

We welcome contributions from the community to enhance OpenNuggetizer's capabilities. Please refer to our contribution guidelines for more information.

**License:**

This project is licensed under the Apache License v2.0.

