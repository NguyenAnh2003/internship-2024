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


URL = [
    "https://www.tripadvisor.com/Attraction_Review-g293916-d450970-Reviews-BTS_Skytrain-Bangkok.html",
    "https://www.quora.com/What-do-you-think-of-the-train-system-in-Bangkok",
    "https://www.quora.com/Why-do-many-say-that-their-train-journey-was-very-smooth-in-Thailand-when-actually-the-trains-in-Thailand-for-the-most-part-arent-that-great",
    "https://www.quora.com/What-is-the-best-way-to-get-around-Bangkok",
    "https://www.quora.com/unanswered/What-is-your-opinion-on-the-train-or-subway-system-in-Bangkok-Do-you-think-it-is-worth-using",
    "https://www.quora.com/Does-Bangkok-have-a-metro-system",
]


options = Options()
options.add_experimental_option("detach", True)

service = Service(executable_path="C:\Program Files\chromedriver.exe")

driver = webdriver.Chrome(service=service, options=options)

driver.get(url=URL[1])

PAUSE_TIME = 2
action = ActionChains(driver)

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


def quora_scraper():
    try:
        jsonfile = open("data.json", "w", encoding="utf-8")

        # button = i.find_element(By.CLASS_NAME, "q-click-wrapper")
        # if button:
        # action.click(on_element=button).perform()

        contents = driver.find_elements(By.CLASS_NAME, "qu-userSelect--text")

        x = 0

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


quora_scraper()
# driver.maximize_window()
