import re
import requests
from selenium.common import NoSuchElementException
from selenium.webdriver import Chrome, ChromeOptions
import time
import datetime

from selenium.webdriver.chrome import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys

import json
import os
import random
import numpy as np
from selenium.webdriver.support.wait import WebDriverWait

from facebook.util.Exception import TException


class SaveData:
    judge = {}
    index = []
    sort = 0
    range_t = 100000

    def __init__(self):
        pass

    # 判断是否是数字
    # 转发，评论，点赞为空时返回0
    @staticmethod
    def is_digit(content):
        return content if content.isdigit() else 0

    # 替换表情和多个空格
    @staticmethod
    def clean_emo(content):
        content = re.sub(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+', '', content)
        return re.sub(u'[\s]{2,}', ' ', content)

    @staticmethod
    def clean_comma(content):
        return re.sub(u'[,]{1,}', ' ', content)

    @staticmethod
    def get_path(file_type):
        today = datetime.date.today()
        hour = datetime.datetime.now().hour
        try:
            # TODO 此处存储为为相对路径 应改为绝对路径
            path1 = 'D:/jiajian/School/shixi/1/cankao/video' + str(today) + '/' + str(hour)
            if not os.path.exists(path1):
                os.makedirs(path1, exist_ok=True)
        except Exception as e:
            print(e)
        file_name = int(time.time() + np.random.randint(0, SaveData.range_t, 1))
        if file_type == 1:
            path = path1 + "/" + str(file_name) + ".mp4"
        elif file_type == 2:
            path = path1 + "/" + str(file_name) + ".jpg"
        return path

    @staticmethod
    def get_data(div, pre_content, drive):
        content_str = ''
        try:
            # 获取用户名
            user_name_t = div.find_element(By.XPATH,
                                           value='//*[@class="xt0psk2"]/a/strong/span').text
            # 获取内容
            content_list = div.find_element(by=By.XPATH,
                                            value='//*[@class="x1iorvi4 x1pi30zi x1l90r2v x1swvt13"]/div/div/span/div')
            for content in content_list:
                content_str += (content.text + " ")
            # 判断是否没有新的内容 如果第17条内容与前16条某一条一致，则认为没有新的内容了，直接返回并进行下一个主题的搜索
            for value in SaveData.judge.values():
                if value == content_str:
                    return TException.NO_NEW_CONTENT

            # todo 引入更多维度的判断 目前默认保存最后16条内容
            if len(SaveData.judge.keys()) < 16:
                SaveData.judge[SaveData.sort] = content_str
                SaveData.sort += 1
            else:
                if SaveData.sort == 16:
                    SaveData.sort = 0
                SaveData.judge[SaveData.sort] = content_str
                SaveData.sort += 1

        except NoSuchElementException as e:
            print(e)
            return TException.STRUCTURE_EXCEPTION
        if pre_content == content_str:
            return TException.NO_NEW_CONTENT
            # elif len(bottom_list) == 0:
            # return TException.STRUCTURE_EXCEPTION

        # 获取点赞数
        approval_num = drive.find_element(By.XPATH,
                                          value='//*[@class="x6s0dn4 x78zum5 x1iyjqo2 x6ikm8r x10wlt62"]/div/span').text
        print("approval_num is ", approval_num)
        # todo 评论数，转发数
        # 返回需数据库保存的一行记录
        return (content_str,
                (
                    user_name_t, SaveData.clean_emo(content_str),
                    # SaveData.is_digit(transmitting_num),
                    SaveData.is_digit(approval_num)
                )
                )


class BrowserOp:
    def __init__(self, url="https://www.facebook.com/"):
        options = ChromeOptions()
        prefs = {
            'profile.default_content_setting_values':
                {'notifications': 2}  # 禁止谷歌浏览器弹出通知消息
        }
        options.add_experimental_option('prefs', prefs)
        self.driver = Chrome(chrome_options=options)
        self.driver.get(url)
        self.driver.maximize_window()
        time.sleep(3)

    def login(self, account, password):
        time.sleep(2)
        self.driver.find_element(By.XPATH,
                                 value='//*[@id="email"]').send_keys(account)
        self.driver.find_element(By.XPATH,
                                 value='//*[@id="pass"]').send_keys(password)
        time.sleep(1)
        self.driver.find_element(By.NAME,
                                 value='login').click()
        time.sleep(3)

    @staticmethod
    def get_config(path="config/config.json"):
        config = ""
        with open(path, mode="r", encoding="utf8") as file:
            config = json.load(file)
        return config

    # 选择最后一个窗口
    def select_new_window(self, index=-1):
        windows = self.driver.window_handles
        self.driver.switch_to.window(windows[index])

    # 搜索关键字
    def search(self, key):
        WebDriverWait(self.driver, 10, 0.5).until(lambda driver: driver.find_element(by=By.XPATH,
                                                                                     value='//input[@aria-label="搜索 Facebook"]'))
        input_f = self.driver.find_element(by=By.XPATH,
                                           value='//input[@aria-label="搜索 Facebook"]')
        input_f.send_keys(Keys.CONTROL, "a")
        input_f.send_keys(Keys.BACKSPACE)
        input_f.send_keys(key)
        input_f.send_keys(Keys.ENTER)

    def tiezi(self):
        self.driver.find_element(by=By.LINK_TEXT,
                                 value='帖子').click()
        time.sleep(1)
        self.select_new_window()
