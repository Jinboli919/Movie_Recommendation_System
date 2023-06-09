# Movie_Recommendation_System
### Introduction
With the increasing availability of movies on various platforms, the challenge of selecting the most preferred movies for users has become a critical issue in the entertainment industry. With so many options to choose from, it can be overwhelming to decide what to watch next. Therefore, the field of recommendation systems has been the subject of important and extensive research in recent years, as more and more companies, such as Netflix and Amazon Prime, developed their own recommendation systems to provide users with a more customized and personalized viewing experience. And this project aims to build a movie recommendation system to produce a ranked list of movies to provide customized and personalized recommendations based on users’ preferences, rating history, and other relevant movies’ features. 

In this project, 3 different kinds of recommendation mechanisms are implemented. First one is content-based filtering, and second one is collaborative filtering. And 3 methods in collaborative filtering are implemented. They are User-Based Collaborative Filtering, Item-Based Collaborative Filtering and Matrix Factorization(MLP Approach and SVD Approach). Then I combined content-based filtering and collaborative filtering to create a hybrid filtering as the third one that takes advantage of the strengths of both methods. By leveraging the power of collaborative filtering and content-based filtering, this system can provide accurate and relevant recommendations that help users discover new movies they might not have otherwise considered. Apart from this, an interactive web page was built. This web page is user-friendly and intuitive, with clear instructions and a simple interface, which allows users to easily access and provides users with an enhanced experience.

### Dataset Description
The datasets in the project are extracted from the official website of IMDB and MovieLens, consisting of over 100k movie rating records from 671 users and including over 45k movies, with a total of 26 columns containing detailed movie information, such as movie description, movie casts, movie keywords, movie overviews and so on etc. There are total 6 csv files. They are movies, ratings, keywords, crew, movies_m and links.

Check out at the Datasets file.

## Coding Part
#### EDA 
EDA.py
#### TOP 50 Movies and top movies in different genres
TOP Movies




The link to the Interactive Web Application of Movie Recommendation System: https://movie-recommendation-system-jinbo-li.streamlit.app/
