import urllib.request
import requests
from bs4 import BeautifulSoup
import os

# Define a variable and put the link on that
html = requests.get("https://en.wikipedia.org/wiki/Deep_learning")
#Parse the source code using the Beautiful Soup library and save the parsed code in a variable
soup = BeautifulSoup(html.content, "html.parser")

#Print out the title of the page
print(soup.title)

#Find all the links in the page
print(soup.find_all('a'))

#Iterate over each tag(above) then return the link using attribute "href" using get
for link in soup.find_all('a'):
    print(link.get('href'))