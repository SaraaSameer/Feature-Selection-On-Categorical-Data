import uvicorn
from fastapi import FastAPI, Query, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import classifier
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer 
from bs4 import BeautifulSoup
import math


app = FastAPI()
templates = Jinja2Templates(directory= 'templates')
app.mount('/static', StaticFiles(directory= 'static'), name = 'static')

data = " "

@app.get('/')
def hello_World():
     return {'Sucesss': 'Sucessfully Loaded'}

@app.get('/basic')
def homepage(request: Request):
    return templates.TemplateResponse('index.html',{'request':request})

@app.post('/basic')
def homepage(request:Request, query: str= Form(...), action:str = Form(...)):
    if action == "classify":
        ps = PorterStemmer()
        CLEANR = re.compile('<.*?>')
        query= re.sub(CLEANR, '', query)
        print(query)
        stemmed_content = re.sub('[^a-zA-Z]',' ',query)   #Dropping all encodings, numbers etc
        stemmed_content = stemmed_content.lower()
        stemmed_content = stemmed_content.split()
        stemmed_content = [ps.stem(word) for word in stemmed_content if not word in stopwords.words('english')]
        stemmed_content = ' '.join(stemmed_content)
        stemmed_content = [stemmed_content]
        print(stemmed_content)
        result = classifier.Predict_Query(stemmed_content)
        result = result[0]    #To convert into str
        print(result)
        return templates.TemplateResponse('index.html', {'request':request, 'data':stemmed_content, 
        'result': result})

    elif action == 'features':
        accuracy_score = str(round(classifier.accuracy,2)) +"%"
        return templates.TemplateResponse('index.html', {'request':request, 'data':accuracy_score})



if __name__ == 'main':
    uvicorn.run(app)
    