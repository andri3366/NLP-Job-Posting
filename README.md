# Fraudulent Job Posting Web Application
Independent project I built to challenge my knowledge of machine learning and apply topics I had learn in level 1 of my [Business Intelligence System Infrastructure](https://www.algonquincollege.com/sat/program/business-intelligence-system-infrastructure/) program at Algonquin College.

## Tech stack
- VS Code
- Python 3.12.7 (used to save space on AWS)
- Streamlit
- AWS EC2
- Docker

## Links
- [NLP classification](https://stackabuse.com/python-for-nlp-sentiment-analysis-with-scikit-learn/)
- NLP Text Preprocessing:
    - [Link 1](https://medium.com/@sachinkc263/nlp-data-cleaning-preprocessing-a-complete-practical-guide-from-raw-text-to-machine-ready-2a75faeb40df)
    - [Link 2](https://www.geeksforgeeks.org/nlp/text-preprocessing-for-nlp-tasks/)
- [Combine Features](https://medium.com/@Paddy_643/ways-to-deal-with-mixed-data-types-8d8fc8ed3fd3)
- [Streamlit Guide](https://www.geeksforgeeks.org/python/a-beginners-guide-to-streamlit/)
- [Amazon EC2 & Docker Deployment](https://www.youtube.com/watch?v=DflWqmppOAg)
- OpenAI API
    - [Link 1](https://www.youtube.com/watch?v=q5HiD5PNuck)
    - [Link 2](https://platform.openai.com/api-keys)
 
## Data
- Data set: [Kaggle](https://www.kaggle.com/datasets/shivamb/real-or-fake-fake-jobposting-prediction)
- Columns:
    - job_id: Unique job ID
    - title: job title from the ad entry
    - location: geographical location of the add
    - department: corporate department
    - salary_range: indicative salary range
    - company_profile: company description
    - requirements: enlisted job requirements
    - benefits: enlisted offered benefits
    - telecommuting: work from home opportunities
    - has_company_logo: if company logo is present in the job posting
    - has_questions: if screening questions are present
    - employment_type: full-time, part-time, etc.
    - required_experience: entry level, intern, etc.
    - required_education: bachelors, masters, etc.
    - industry: IT, Real Estate, etc.
    - function: consulting, engineering, etc.
    - fraudulent: classification attribute, 0 for true

## Exploratory Data Analysis & Data Cleaning
  - Load dataset and save to a dataframe
  - Apply basic EDA such as describe, info, sample, head, tail, correlation matrix, etc.
  - Determine if there are any duplicated and missing values
  - Handle missing values and fill all to 'Missing'
      - ```df.isnull().sum()```
      - Observed that all missing values were for object dtypes only
  - Got count of values using the "," as a delimeter for the location column
      - ```df.location.str.count(",").value_counts()```
  - Check if first value on the right is a short form of country (i.e. UK, US)
      - ```df[(df.location.str.count(",") == 10) & (df.location != "Missing")].sample(1)```
  - Get the count of unique values for feature engineering, based on the categorical columns
      - These include employment_type, required_education, required_experience, industry, function
      - The unique values kept for feature engineering were employment_type, required_experience, and required_education as the max unique categories was 14

### Data Visualization
- Inspected employment type, experiences, and education

1. Which employment type is most likely to be fraudulent?
    - The distrabution of employment type for fraudulent cases has the largest chance of scam recruiters targeting flexible workers

2. Which employment type contributes the most fraudulent postings overall?
    - The count of employment type for fraudulent and not fraudulent indicates that full time dominates most of the dataset. There is evidence of high imbalance in the dataset. Eventhough full-time positings have the highest count of fraudulent job postings, part-time and missing have a higher proportion of fraud relative to the total number of postings
  
3. Which education is most likely to be fraudulent?
    - 'Some High School Coursework' had the highest percentage of fraudulent cases, however the count of values was only 27. 'Certification' has the second highest percentage of fraudulent cases, however had only 170 counts. Noteable 'Some College Coursework Completed' has only 102 cases total, however has approximately 5% of the cases being fraudulent.

4. Which required experience is most likely to be fraudulent?
    - The 'Executive' position has the highest percentage of fraudlent cases, however had only 141 cases total. 'Entry Level' however had 2697 total cases, approximately 6.6% were fraudulent. Noteably 'Missing' had 7050 cases total, with approximately 6.3% being fraudulent.
    - This indicates that the 'Executive' position has a relatively high risk of being fraudulent. The 'Entry Level' position has a larger practical impact because of it's volume. Finally, the 'Missing' cases could link to poor-quality postings or intentionally vague fraudulent postings.
  
5. Is there a correlation between work from home and fraudulent cases?
    - The data showed a total of 17113 cases of not telecomutting and 767 cases of telecommuting. Approximately 8% of not telecommuting cases were fraudulent and approximately 4.7% of not telecommuting were fraudulent.
    - This suggests in this dataset that non-telecimmuting jobs are more likely to be fraudulent, which does not support the assumption that work from home postings are more likely to be fraudulent. However, due to the imbalance of the dataset other features are likely to influence this factor.
  
6. Which countries are most likely to be fraudulent?
   - Decided to group each country into top 10. It was decided to extract the top 9 countries based on the total count in the dataset, then group all other countries into the cateogry 'Other' to make a top 10.
   - ```df['country'] = df.location.str.split(',').str[0].replace('', 'Missing').fillna('Missing')```
    - Used the earlier analysis of the number of comma delimeters to determine if the first index of the string was used for only country. Once determined any value in the first index that was missing a short form of a country was put into the 'Missing' cateogry.
    - All values were assigned to a list and a lambda expression was used to place any country not in the original top 9 to the 'Other' cateogry.
    - A heatmap was applied to determine the percentage of fraudulent and not fraudulent cases before category. Australia (AU) had the highest with 19% fraudulent cases, with a total count of 214 cases. New Zealand (NZ) and Greece (GR) both had 0% fraudulent cases, NZ had a total count of 333 and GR had 940 cases. The US had the highest count of cases being 10656 with only 7% being fraudulent. This indicates that 'AU' has the highest risk of being fraudulent, US had the highest overall impact due to the number of listings, and NZ/GR had no fraudulent cases but should still be cautiously observed.
  
### Modeling
1. Created a preprocessing py file to clean the text in preporation for TF-IDF vectorization. Based on my research it is important to preprocess text to prevent poor model performances due to too much noise.
2. Selected Logistic Regression, Random Forest, XGBoost, and Naive Bayes for modeling.
3. Created a training file that combines all the text into one, applies TF-IDF vectorization, handled binary and categorical features. Trained each model along with a different test split to determine the best one. Used F1 score as the evaluation metric.
   - I had choosen F1 score as the metric due to the dataset being imbalanced. Using the harmonic mean of both precision and recall, I could ensure that the model performed well on both fraudulent and non fraudulent listings simultaneously
4. Decided to also apply a text only model just using the TF-IDF vectorization.
5. XGBoost performed the best for the full model and the text model. The split for the text only model resulted in an 80/20 and the full model used a split of 0.15
6. Loaded each model into pkl file.
7. Created a predict function for the full model and text model to be called from the Streamlit app.
   - The full model required the use of the scipy.sparse library to handle the binary and categorical features. The library required numeric so I converted the binary values to integers. For XGBoost models I needed to sparse the values as the model often requires consistent sparse types, I accomplished this by using the csr_matrix function.
   - Used the scipy.sparse hstask function to combine the text, binary, and classification features into one matrix.
  
### Streamlit App and OpenAI
- Created the basic interface for the web application
- Applied functions with ```@st.cache_resource``` to load and reuse the necessary pkl files to speed up the web application
- Implemented ```st.session_state``` to maintain prediction results and user input when generating the LLM explanations
- Tested the API connection to OpenAI using ```$env:OPEN_API_KEY=``` in my PowerShell terminal
- When a user inputs their prompt a py file with an explain_prediction function

### AWS and Docker
- Followed the tutorial video on how to deploy a streamlit web application to Docker
- Code to update website / docker and clear instance space:
```
docker stop $(docker ps -aq)
```
```
docker rm $(docker ps -aq)
```
```
docker system prune -a -f --volumes
```
```
docker system df
```
```
git pull "repo"
```
```
docker build -t "docker_acount/image_name" .
```
```
docker push "docker_account/image_name":latest
```
```
docker pull "docker_account/image_name:latest
```
```
docker run -d -p 8501:8501 \
-e OPENAI_API_KEY="api_key_name" \
"docker_account/image_name":latest
```
### Future Improvements
- Potentially integrate with Snowflake to save user inputs
- Look into applying SBERT for classification.
