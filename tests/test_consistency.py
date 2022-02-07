import sys
import os
from time import sleep

import pytest
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC


# opts = Options()
# opts.headless = True
# testing = webdriver.Edge(opts=opts)


def test_browser(client, chrome, capsys):
    browser = chrome
    URL = "http://127.0.0.1:5000/"
    browser.delete_all_cookies()
    browser.get(URL)
    browser.maximize_window()
    assert browser.title == "Emotive"
    # SKIP: Do not need to check if can register as registration function will be disabled

    # Check that we can log in
    browser.find_elements_by_link_text("Login")[0].click()

    USERNAME = "jane"
    PASSWORD = "Password1234!"

    username_text_field = browser.find_element_by_id("username")
    username_text_field.send_keys(USERNAME)

    password_text_field = browser.find_element_by_id("password")
    password_text_field.send_keys(PASSWORD)
    sleep(5)

    browser.find_element_by_id("submit").click()
    sleep(5)

    assert browser.current_url == "http://127.0.0.1:5000/predict"
