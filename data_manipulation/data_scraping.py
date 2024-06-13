from bs4 import BeautifulSoup
import requests

# URL
_URL = [""]


def scraptor():
	page = [] # paging
 
	for i in enumerate(_URL):
		page = requests.get(_URL[i])

	soup = BeautifulSoup(page.content, "html.parser") # define soup	
	return soup

	