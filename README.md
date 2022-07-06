# Disaster Response Pipeline Project

## Project Motivation

This project analyzes the disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages. This project is helpful for an emergency worker who can access it via web app included as part of the project.

## Project Structure

      .
      ├─ app
      │   ├─ run.py                           | Flask file that runs app
      │   └─ templates
      │       ├─ go.html                      | Classification result page
      │       └─ master.html                  | Master page of web app
      ├─ data
      │   ├─ disaster_categories.csv          | Disaster Categories Dataset
      │   ├─ disaster_messages.csv            | Disaster Messages Dataset
      |   ├─ DisasterResponse.db              | Disaster Response Database to be used by the Web-App
      │   └─ process_data.py                  | Data cleaning
      ├─ models
      │   └─ train_classifier.py              | Train ML model
      │   └─ classifier.pkl                   | Python Pickle File
      └─ README.md

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:4111/
