# Final Project: MLOps Machine Learning API

## Project Overview

This repository contains my final project for the UC Berkeley MIDS Machine Learning Operations (MLOps) course. The goal of this project is to design, build, and deploy a production-ready machine learning API that serves NLP predictions using modern MLOps tools and best practices.

The project demonstrates the full lifecycle of a machine learning service, from model packaging and API development to containerization, orchestration, caching, testing, and monitoring. It is intended to reinforce key concepts in software engineering, cloud deployment, and scalable ML systems.

---

## Learning Objectives

- **Model Packaging:** Use Poetry to manage dependencies and package an NLP model (DistilBERT from Hugging Face) for efficient CPU-based inference.
- **API Development:** Build a FastAPI application to serve predictions from user requests.
- **Testing:** Write and execute pytest tests to ensure application correctness and reliability.
- **Containerization:** Use Docker to package the application for consistent deployment.
- **Caching:** Integrate Redis to cache results and protect the API from abuse.
- **Cloud Deployment:** Deploy the application to Azure Kubernetes Service (AKS) using Kubernetes and Kustomize.
- **Load Testing & Monitoring:** Use K6 for load testing and Grafana for real-time monitoring and visualization of system performance.
- **Security & Best Practices:** Follow secure coding, modular architecture, and reproducibility standards throughout the project.

---

## Folder Structure

```
MLOPS/
├── README.md                    # Project documentation (this file)
├── build-deploy.sh              # Deployment automation script
├── grader.sh                    # Grading/test script
├── images/                      # Diagrams and Grafana screenshots
├── project/
│   ├── Dockerfile               # Container definition
│   ├── docker-compose.yml       # Local orchestration
│   ├── example.py               # Example usage
│   ├── load.js                  # K6 load testing script
│   ├── poetry.lock, pyproject.toml # Dependency management
│   ├── README.md                # API documentation
│   ├── .k8s/                    # Kubernetes manifests (bases, overlays)
│   ├── src/                     # Source code (main.py, __init__.py)
│   ├── tests/                   # Unit tests
│   ├── yamls/                   # Additional Kubernetes configs
├── trainer/
│   └── train.py                 # Model training script
```

---

## Methodology

1. **Model Training & Packaging:**  
   - Train a sentiment analysis model using DistilBERT and push it to Hugging Face.
   - Package the model for efficient inference in the API.

2. **API Development:**  
   - Implement a FastAPI service with Pydantic models for input validation.
   - Serve predictions via REST endpoints.

3. **Testing & Validation:**  
   - Write pytest unit tests to verify API functionality and edge cases.

4. **Containerization & Orchestration:**  
   - Build Docker images and use docker-compose for local development.
   - Deploy to Azure Kubernetes Service using Kustomize and Kubernetes manifests.

5. **Caching & Security:**  
   - Integrate Redis for result caching and endpoint protection.
   - Follow secure coding practices and avoid hardcoding secrets.

6. **Load Testing & Monitoring:**  
   - Use K6 to simulate user load and Grafana to visualize system metrics.

---

## Getting Started

1. Clone the repository.
2. Review the documentation and code in the `project/` folder.
3. Follow the setup instructions in `README.md` and `build-deploy.sh` to run and test the API locally or on Azure.
4. Explore the `trainer/` folder for model training details.

---

## Contributors

- Victor Ramirez
- UC Berkeley MIDS MLOps Course

---

Main Project Repository: [datasciw255](https://github.com/vhr1975/datasciw255)

---

## License

This project is for educational purposes as part of the UC Berkeley MIDS program.