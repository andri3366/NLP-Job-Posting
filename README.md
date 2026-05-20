# Fraudulent Job Posting Web Application
Independent project I built to challenge my knowledge of machine learning and apply topics I had learn in level 1 of my [Business Intelligence System Infrastructure](https://www.algonquincollege.com/sat/program/business-intelligence-system-infrastructure/) program at Algonquin College.

### Tech stack
- VS Code
- Python 3.12.7 (used to save space on AWS)
- Streamlit
- AWS EC2
- Docker

### Links
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
 
### Data
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

  ### Exploratory Data Analysis & Data Cleaning
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
      - 
