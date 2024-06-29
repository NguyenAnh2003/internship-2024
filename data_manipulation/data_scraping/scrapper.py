from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time
from selenium.common.exceptions import TimeoutException, NoSuchElementException
import requests
from selenium.webdriver.common.action_chains import ActionChains
import json
from urls import QUORA_URL



jsonfile = open("data.json", "w", encoding="utf-8")

options = Options()
options.add_experimental_option("detach", True)

service = Service(executable_path="C:\Program Files\chromedriver.exe")

driver = webdriver.Chrome(service=service, options=options)
x = 0


def quora_scraper():
    global x
    try:
        contents = driver.find_elements(By.CLASS_NAME, "qu-userSelect--text")

        for p in contents:
            paragraphs = p.find_elements(By.CLASS_NAME, "qu-wordBreak--break-word")
            for i, item in enumerate(paragraphs):
                if item.text == "":
                    continue
                elif item.text == "Continue Reading":
                    continue
                print(f"text: {item.text}")
                data = {f"feedback {x}": item.text}

                json.dump(data, jsonfile, ensure_ascii=False)
                jsonfile.write("\n")

                x += 1

        # contents = WebDriverWait(driver, 5).until(
        #     EC.presence_of_all_elements_located((By.CLASS_NAME, "qu-userSelect--text"))
        # )

        # for p in contents:
        #     paragraphs = p.find_elements(By.CLASS_NAME, "qu-wordBreak--break-word")
        #     for i, item in enumerate(paragraphs):
        #         if item.text == "":
        #             continue
        #         elif item.text == "Continue Reading":
        #             continue
        #         print(f"text: {item.text}")
        #         data = {f"feedback {i}": item.text}

        #         json.dump(data, jsonfile, ensure_ascii=False)
        #         jsonfile.write("\n")
    except Exception as e:
        raise ValueError(e.__cause__)


visited_urls = set()

for i, item in enumerate(QUORA_URL):
    if item not in visited_urls:
        driver.get(url=item)

        PAUSE_TIME = 2
        lh = driver.execute_script("return document.body.scrollHeight")

        while True:
            try:
                driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
                time.sleep(PAUSE_TIME)
                nh = driver.execute_script("return document.body.scrollHeight")
                if nh == lh:
                    break
                lh = nh
            except NoSuchElementException as e:
                raise ValueError(e)

        quora_scraper()
        
        visited_urls.add(item)

# driver.maximize_window()
