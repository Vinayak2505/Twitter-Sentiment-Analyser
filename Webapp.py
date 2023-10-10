#Importing Dependencies
import pandas as pd
import re      #For cleaning obtained tweets 
import joblib  #For saving pipeline

import selenium           #For getting tweets from Twitter page
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
from selenium import webdriver
from time import sleep    #To wait for the page to get loaded
import translators as ts  #Translate tweets to English

import matplotlib.pyplot as plt #To plot results
import nltk
import math

from flask import Flask, render_template, redirect, url_for, request #For creating webapp


def get_tweets(query,n):
    #Setting up login
    PATH = "chromedriver.exe"
    driver = webdriver.Chrome(PATH)
    driver.get("https://twitter.com/login")
    sleep(10)

    #Logging into a demo twitter account
    username = driver.find_element(By.XPATH, "//input[@name='text']")
    username.send_keys("swarupps66@gmail.com")
    next_button = driver.find_element(
        By.XPATH, "//span[contains(text(),'Next')]")
    next_button.click()
    sleep(5)

    #Checking if unusual activity page has been returned
    try:
        check = driver.find_element(By.XPATH, "//input[@name='text']")
        check.send_keys("SwaruppS")
        driver.find_element(
            By.XPATH, "//span[contains(text(),'Next')]").click()
    except:
        f = 1
    sleep(10)
    #Enter password for demo account
    pwd = driver.find_element(By.XPATH, "//input[@name='password']")
    pwd.send_keys("commonemail")
    login = driver.find_element(By.XPATH, "//span[contains(text(),'Log in')]")
    login.click()
    sleep(10)

    #Search account with given username
    search = driver.find_element(
        By.XPATH, "//input[@data-testid='SearchBox_Search_Input']")
    search.send_keys(query)
    search.send_keys(Keys.ENTER)
    sleep(10)

    #Moving to peoples' tab
    people = driver.find_element(By.XPATH, "//span[contains(text(),'People')]")
    people.click()
    sleep(10)
    query = "//span[contains(text(),'@"+query+"')]"
    profile = driver.find_element(
        By.XPATH, query)
    profile.click()

    #Function to clean the translated tweets
    def remove_tags(string):
        result = re.sub('','',string)                 #Remove HTML tags
        result = re.sub('\n', ' ', result)            #Remove next line characters in a tweet
        result = re.sub(',', '', result)              #Remove commas in a tweet
        result = re.sub('https://.*','',result)       #Remove URLs
        result = re.sub(r'[^a-zA-Z0-9]', ' ', result) #Remone non-alpha numeric characters
        result = result.lower()                       #Ronvert to lower case
        return result

    #It may create a problem when we store tweets having a comma or next line characters
    #Function to clean original tweets
    def rt(string):
        result = re.sub('\n', ' ', string)    #Remove next line characters in a tweet
        result = re.sub(',', '', result)      #Remove commas in a tweet
        return result
    
    tweets = ""
    scroll = 0
    a = 0
    unq = ""
    #Getting tweets while scrolling through the person's account
    while a < n:
        try:
            tweet = driver.find_element(
                By.XPATH, "//div[@data-testid='tweetText']").text
            if driver.find_element(By.XPATH, "//div[@data-testid='tweetText']").get_attribute('id') != unq:
                tweets += rt(tweet) + "," + remove_tags(ts.translate_text(
                    tweet, translator='google')) + "\n"
                a += 1
                unq = driver.find_element(
                    By.XPATH, "//div[@data-testid='tweetText']").get_attribute('id')
                driver.execute_script('window.scrollBy(0,150);')
                sleep(1)
            else:
                driver.execute_script('window.scrollBy(0,150);')
                sleep(1)
        except:
            driver.execute_script('window.scrollBy(0,150);')
            sleep(1)
    driver.close()
    #Store the original and translated tweets in a csv file
    file = open("tweet.csv", "w", encoding="utf-8")
    file.write("tweet,tr_tweet\n")
    file.write(tweets)
    file.close()


app = Flask(__name__)
app.static_folder = 'static'


@app.route('/')
def home():
    return render_template("index.html", X="")


@app.route("/predict", methods=['POST', 'GET'])
def pred():
    #try:
    if request.method == 'POST':
        #Get the username entered
        query = request.form['query']
        #Get the number of tweets to be scraped
        n = int(request.form['no_tweets'])
        #Function to store the tweets of that user in a csv file
        get_tweets(query,n)
        #Get tweets
        pdata = pd.read_csv(
            'tweet.csv', encoding='utf-8', on_bad_lines='skip', encoding_errors='ignore')
        #Load the saved pipeline
        nb_classifier = joblib.load('SentimentPredictor.pkl')
        #Predict sentiments of the obtained tweets
        pdata['sentiment'] = nb_classifier.predict(pdata['tr_tweet'])
        
        #Count number of positive, negative and neutral tweets
        try:
            pos = pdata['sentiment'].value_counts()['Positive']
        except:
            pos = 0
        try:
            neu = pdata['sentiment'].value_counts()['Neutral']
        except:
            neu = 0
        try:
            neg = pdata['sentiment'].value_counts()['Negative']
        except:
            neg = 0

        #Store predicted data to a csv file
        pdata.to_csv('results.csv')

        #Plot the results
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [pos, neu, neg]
        colors = ['green', 'gold', 'red']
        plt.bar(x=labels, height=sizes, width=0.6, color=colors)
        plt.xlabel("Sentiment of tweets")
        plt.ylabel("No. of tweets with sentiment")
        plt.title('Sentiment Analysis Results')
        plt.savefig('static/image.png')
    return render_template('result.html', value=pdata['tweet'], senti=pdata['sentiment'], size=range(len(pdata.index)))
    #except:
        #return render_template('index.html', X="Error")


if __name__ == '__main__':
    app.debug = True
    app.run(host='localhost')