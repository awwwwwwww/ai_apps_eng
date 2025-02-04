# AI Applications in Engineering - Example Capstone Project

## Project Title: Recipe Recommender

### Overview
This repository contains an example capstone project for the AI Applications in Engineering course. The project focuses on using machine learning techniques to predict equipment failures and optimize maintenance schedules, improving operational efficiency and reducing downtime in industrial settings.

### Objectives
- Develop a predictive model for equipment failure based on sensor data.
- Demonstrate data preprocessing, feature engineering, and model selection techniques.
- Deploy the trained model in a simulated production environment.
- Evaluate model performance and discuss its real-world implications.

### Dataset
The project uses a publicly available dataset, which includes:
- recipes
- ingredients 

### Technologies Used
- **Python** (primary programming language)
- **Jupyter Notebook** (for development and visualization)
- **Pandas & NumPy** (for data manipulation)
- **Scikit-learn & TensorFlow/PyTorch** (for model development)
- **Flask/FastAPI** (for model deployment)
- **Docker** (for containerization)


### Project Structure
```plaintext
├── data/                 # Raw and processed datasets
├── notebooks/            # Jupyter notebooks for EDA and modeling
├── src/                  # Source code for model training and deployment
│   ├── preprocessing.py  # Data cleaning and feature engineering
│   ├── train.py          # Model training script
│   ├── verify.py      # Model verification script
├── deployment/           # Deployment-related files (Docker, API, etc.)
├── reports/              # Project reports and presentations
├── README.md             # Project documentation
```

### Setup & Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/example/ai-capstone-project.git
   ```
2. Navigate to the project directory:
   ```sh
   cd ai-capstone-project
   ```
3. Create and activate a virtual environment:
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
4. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```

### Running the Project
- **Exploratory Data Analysis (EDA):** Run the notebooks in `notebooks/`
- **Training the Model:** Execute `python src/train.py`
- **Deploying the Model:** Start the API with `python src/inference.py`
- **Monitoring Performance:** Launch the monitoring stack with Docker:
  ```sh
  docker-compose up -d
  ```

### Expected Outcomes
- A trained AI model capable of predicting equipment failures with high accuracy.
- A deployed inference API for real-time predictions.
- A dashboard displaying key performance metrics and insights.

### Contributing
Students are encouraged to improve and extend this project. Contributions may include:
- Enhancing the dataset with additional features
- Implementing alternative ML models for comparison
- Improving the deployment pipeline

### License
This project is licensed under the MIT License.

---

For any questions, please contact the course instructor or open an issue in the repository.

