"""
Script for using Google Search as distant supervision,alternative to Probase. Probase is more useful
"""
from googlesearch.googlesearch import GoogleSearch

response = GoogleSearch().search("denim")
flag = False
for result in response.results:
    print("Title: " + result.title)
    text = result.getText().lower()
    title  = result.title.lower()
    print text



