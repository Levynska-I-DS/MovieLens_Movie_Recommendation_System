![](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*t98V5s6uNKVNEde5ZYQemw.jpeg)

# MovieLens Movie Recommendation System

### Project Overview

**Objective:** The goal of this project is to build a movie recommendation system using machine learning techniques. The system leverages the MovieLens dataset to predict user ratings for movies and provide personalized recommendations.

**Context:** Building a recommendation system involves handling large datasets and complex algorithms to predict user preferences accurately. Challenges include dealing with sparse data, ensuring scalability, and optimizing the recommendation algorithms for accuracy and performance.

**Significance:** Accurate movie recommendations can enhance user experience by providing personalized content, which is crucial for streaming services and content platforms. This project aims to improve the relevance of recommendations and can benefit further research in collaborative filtering and matrix factorization techniques.

**Goal:** The primary aim is to develop a robust recommendation system that can predict user ratings and suggest movies based on past interactions. The project also explores different models and parameters to enhance prediction accuracy and efficiency.

## Team Members

- **Iryna:** [GitHub](https://github.com/Levynska-I-DS)
- **Christian:** [GitHub](https://github.com/Kriss-fullstack)
- **Semih:** [GitHub](https://github.com/semihd97)

## Jupyter Notebooks

This project consists of several Jupyter Notebooks, each serving different purposes:

1. **Data_Preprocessing.ipynb:** Covers initial data exploration and preprocessing steps, including loading the MovieLens dataset, handling missing values, performing exploratory data analysis (EDA), and preparing data for model training.

2. **Model_Training.ipynb:** Focuses on training various recommendation models, including collaborative filtering and matrix factorization. Includes hyperparameter tuning and model evaluation to select the best-performing model.

3. **Evaluation_and_Predictions.ipynb:** Evaluates the performance of trained models using metrics such as RMSE and MAE. Includes code for generating movie recommendations for users based on the final model.

4. **Data_Enrichment_and_Model_Improvement.ipynb:** Demonstrates the development of an enhanced movie recommendation system using the MovieLens dataset enriched with data from The Movie Database (TMDB) API. 

5. **Interface_method_model_ML_GUI.ipynb:** Provides a detailed description of the graphical user interface (GUI) that interacts with the machine learning model. Explains the GUI's purpose, how to run it from a Python script, and its integration with the recommendation system.

6. **PowerBI_Analysis.ipynb:** Includes Power BI integration for visualizing and analyzing the recommendation system's performance and insights. Helps in understanding data trends and evaluating model effectiveness through interactive dashboards.

7. **Normal_Predictor_Model.ipynb:** Demonstrates the implementation of a basic recommendation model using the Normal Predictor from the Surprise library. This model serves as a baseline, making random predictions based on the distribution of ratings.

8. **KNN_Model.ipynb:** Implements a K-Nearest Neighbors (KNN) model for movie recommendation. Evaluates the KNN algorithm's performance and compares it with other recommendation models.

## Installation and Setup

To set up the project locally, follow these steps:

1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/MovieLens-Recommendation-System.git
    ```

2. Navigate to the project directory:
    ```bash
    cd MovieLens-Recommendation-System
    ```

3. Install and create the conda environment using the provided `environment.yml` file:
    ```bash
    conda env create --file environment.yml
    ```
   This will create a new conda environment named `movie_rec_sys` (or whatever name you specified in the `environment.yml` file).

4. Activate the conda environment:
    ```bash
    conda activate movie_rec_sys
    ```

5. (Optional) If you're using Visual Studio Code, you can select the newly created environment as your Python interpreter.

6. Download the MovieLens dataset and place it in the `data` directory. The dataset can be acquired from [MovieLens](https://grouplens.org/datasets/movielens/).

**Note:** Ensure that you have Anaconda or Miniconda installed on your system before following these steps.

Once the setup is complete, you can use the provided Jupyter Notebooks to preprocess data, train models, and make predictions.

## Dataset

The dataset used in this project is the MovieLens dataset, which contains movie ratings provided by users. The data includes user ratings, movie metadata, and timestamps. You can download the dataset from [MovieLens](https://grouplens.org/datasets/movielens/).

## Attribute Information

The dataset contains the following attributes:

1. **userId:** Identifier for the user.
2. **movieId:** Identifier for the movie.
3. **rating:** User rating for the movie (ranging from 0.5 to 5.0).
4. **timestamp:** Time when the rating was given.

The dataset contains a total of 33,832,162 ratings and 86,537 movies. The ratings dataset includes all individual ratings, while the movies dataset provides metadata for each movie.

## EDA/Cleaning

The initial exploratory data analysis (EDA) includes visualizing the distribution of ratings, understanding the sparsity of the user-movie matrix, and identifying any data quality issues. Data cleaning steps involve handling missing values, normalizing ratings, and splitting the data into training and testing sets.

## Model Choices

We evaluated several recommendation models:

- **SVD (Singular Value Decomposition):** A matrix factorization technique that approximates user-item interactions by decomposing the rating matrix.
- **NMF (Non-negative Matrix Factorization):** Another matrix factorization method focused on non-negative values, offering a different approach to factorization.
- **BaselineOnly:** A basic model that accounts for user and item biases without additional complexity.
- **KNN (K-Nearest Neighbors):** A neighborhood-based approach that makes recommendations based on the similarity between users or items.
- **Normal Predictor:** A baseline model making random predictions based on the distribution of ratings.

The models were evaluated using RMSE (Root Mean Square Error) and MAE (Mean Absolute Error). The final evaluation results are as follows:

| Model            | RMSE   | MAE   |
|------------------|--------|-------|
| **BaselineOnly** | 0.8639 | 0.6583|
| **SVD**          | 0.9004 | 0.6880|
| **NMF**          | 0.9601 | 0.7301|
| **KNN**          | 1.0689 | 0.8447|
| **Normal Predictor** | 1.4495 | 1.1542|

The `BaselineOnly` model was found to be the best-performing model with the lowest RMSE and MAE. This model was also used to test enriched data from The Movie Database (TMDB) API.

## Data Enrichment and Model Improvement

To enhance the recommendation system, we enriched the dataset with additional data from The Movie Database (TMDB) API. The process involved:

1. Fetching additional data from TMDB API for the first 2000 movies.
2. Normalizing TMDB ratings and merging them with our MovieLens data.
3. Retraining the `BaselineOnly` model on this enriched dataset.

### Results of Enriched Model

Our enriched `BaselineOnly` model achieved:
- RMSE on test set: 0.8634
- MAE on test set: 0.6575


### Interpretation

The enriched `BaselineOnly` model showed slight improvements in performance with the enriched data. It outperformed all other models, including SVD and NMF. This suggests that while the TMDB data enrichment provided a small boost, it enhanced the overall recommendation quality.

## Results

Our BaselineOnly-based recommendation system, even with enriched data, proves to be robust and effective for movie recommendations. The data enrichment process demonstrates the potential for integrating external data sources to enhance recommendation systems, although in this case, the improvement was minimal.

Future work could explore more sophisticated ways of integrating external data or experimenting with hybrid models to further improve performance.

## Final Remarks

This project demonstrates the application of machine learning techniques to build a recommendation system using the MovieLens dataset. Future work may involve incorporating additional features, exploring deep learning approaches, or deploying the recommendation system as a web application for real-time use.



