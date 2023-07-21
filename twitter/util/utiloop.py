import re
from selenium.common import NoSuchElementException
from selenium.webdriver import Chrome, ChromeOptions
import time
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium import webdriver
import json
from twitter.util.Exception import TException
from selenium.webdriver.support.ui import WebDriverWait


class SaveData:
    judge = {}
    index = []
    sort = 0

    def __init__(self):
        pass

    # 判断是否是数字
    # 转发，评论，点赞没有时返回0
    @staticmethod
    def is_digit(content):
        if content == 0:
            return content
        return content if content.isdigit() else 0

    # 替换表情和多个空格
    @staticmethod
    def clean_emo(content):
        content = re.sub(u'['u'\U0001F300-\U0001F64F' u'\U0001F680-\U0001F6FF'u'\u2600-\u2B55]+', '', content)
        return re.sub(u'[\s]{2,}', ' ', content)

    @staticmethod
    def clean_comma(content):
        return re.sub(u'[,]{1,}', ' ', content)

    # 获取数据
    @staticmethod
    def get_data(div, pre_content, driver, length):
        content_str = ""
        try:
            # 获取用户名
            user_name_t = div.find_element(By.XPATH,
                                           value='.//span[@class="css-901oao css-16my406 css-1hf3ou5 r-poiln3 r-bcqeeo r-qvutc0"]/span').text
            # 获取推文时间
            time_elements = div.find_elements(By.XPATH,
                                              '//*[@class="css-4rbku5 css-18t94o4 css-901oao r-14j79pv r-1loqt21 r-xoduu5 r-1q142lx r-1w6e6rj r-37j5jr r-a023e6 r-16dba41 r-9aw3ui r-rjixqe r-bcqeeo r-3s2u2q r-qvutc0"]//time')
            time = time_elements[length].get_attribute('datetime')
            print("time is ", time)
            # 获取推文内容
            content_list = div.find_elements(By.XPATH,
                                             value='.//div[@class="css-1dbjc4n"]/div[1]/span')
            # 内容包含多个span 循环获取内容
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
            # 获取转发 评论 点赞 浏览量标签
            bottom_list = div.find_elements(By.XPATH,
                                            #
                                            value='.//div[@class="css-1dbjc4n"]/div[@class="css-1dbjc4n r-1kbdv8c '
                                                  'r-18u37iz r-1wtj0ep r-1s2bzr4 r-1mdbhws" and @id]/div')
        except NoSuchElementException as e:
            print(e)
            return TException.STRUCTURE_EXCEPTION
        if pre_content == content_str:
            return TException.NO_NEW_CONTENT
        elif len(bottom_list) == 0:
            return TException.STRUCTURE_EXCEPTION
        # 获取 评论数，转发数，点赞数，浏览量
        try:
            # 评论数
            comment_num = bottom_list[0].find_element(By.XPATH, value='.//span/span/span').text
        except NoSuchElementException:
            comment_num = 0
        print("comment_num is ", comment_num)

        try:
            # 转发数
            transmitting_num = bottom_list[1].find_element(By.XPATH, value='.//span/span/span').text
        except NoSuchElementException:
            transmitting_num = 0
        print("transmitting_num is ", transmitting_num)

        try:
            # 点赞数
            approval_num = bottom_list[2].find_element(By.XPATH, value='.//span/span/span').text
        except NoSuchElementException:
            approval_num = 0
        print("approval_num is ", approval_num)

        try:
            # 浏览量
            view_num = bottom_list[3].find_element(By.XPATH, value='.//span/span/span').text
        except NoSuchElementException:
            view_num = 0
        print("view_num is ", view_num)
        # 返回需数据库保存的一行记录
        return (content_str,
                (
                    user_name_t, time, SaveData.clean_emo(content_str), SaveData.is_digit(comment_num),
                    SaveData.is_digit(transmitting_num),
                    SaveData.is_digit(approval_num), view_num
                )
                )


class BrowserOp:
    def __init__(self, url="https://twitter.com"):
        options = ChromeOptions()
        options.add_experimental_option("detach", True)
        self.driver = Chrome(chrome_options=options)
        self.driver.get(url)
        self.driver.maximize_window()
        time.sleep(3)

    def login(self, account, password):
        WebDriverWait(self.driver, 10, 0.5).until(lambda driver: driver.find_element(By.XPATH,
                                                                                     value='//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div'
                                                                                           '/div/div[2]/div[2]/div/div/div/div[5]/label/div/div[2]/div/input'))

        self.driver.find_element(By.XPATH,
                                 value='//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/'
                                       'div/div[2]/div[2]/div/div/div/div[5]/'
                                       'label/div/div[2]/div/input').send_keys(account)
        self.driver.find_element(By.XPATH,
                                 value='//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/'
                                       'div/div/div[2]/div[2]/div/div/div/div[6]/div').click()

        WebDriverWait(self.driver, 10, 0.5).until(lambda driver: driver.find_element(By.XPATH,
                                 value='//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/di'
                                       'v[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input'))
        self.driver.find_element(By.XPATH,
                                 value='//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div/div[2]/di'
                                       'v[2]/div[1]/div/div/div[3]/div/label/div/div[2]/div[1]/input').send_keys(password)
        self.driver.find_element(By.XPATH,
                                 value='//*[@id="layers"]/div/div/div/div/div/div/div[2]/div[2]/div/div'
                                       '/div[2]/div[2]/div[2]/div/div[1]/div/div/div/div').click()
        time.sleep(2)
        #BrowserOp.browser_sleep(5)

    @staticmethod
    def get_config(path="config/config.json"):
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
                                           value='//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]'
                                                 '/div/div[1]/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/'
                                                 'div/div/form/div[1]/div/div/div/label/div[2]/div/input'))
        input_f = self.driver.find_element(by=By.XPATH,
                                           value='//*[@id="react-root"]/div/div/div[2]/main/div/div/div/div[1]'
                                                 '/div/div[1]/div[1]/div[1]/div/div/div/div/div[1]/div[2]/div/'
                                                 'div/div/form/div[1]/div/div/div/label/div[2]/div/input')
        input_f.send_keys(Keys.CONTROL, "a")
        input_f.send_keys(Keys.BACKSPACE)
        input_f.send_keys(key)
        input_f.send_keys(Keys.ENTER)
        time.sleep(5)
        #BrowserOp.browser_sleep(10)

    # 点击explore
    def click_explore(self):

        WebDriverWait(self.driver, 10, 0.5).until(lambda driver: driver.find_element(by=By.XPATH,
                                 value='//*[@id="react-root"]/div/div/div[2]/header/'
                                       'div/div/div/div[1]/div[2]/nav/a[2]/div[1]'))

        self.driver.find_element(by=By.XPATH,
                                 value='//*[@id="react-root"]/div/div/div[2]/header/'
                                       'div/div/div/div[1]/div[2]/nav/a[2]/div[1]').click()
        time.sleep(1)
        #BrowserOp.browser_sleep(2)

    # 点击下一页
    def next_page(self, num):
        self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        time.sleep(1)
        if num == 2:
            self.driver.find_element(by=By.XPATH, value='//*[@id="pl_feedlist_index"]/div[5]/div/a').click()
        else:
            self.driver.find_element(by=By.XPATH, value='//*[@id="pl_feedlist_index"]/div[5]/div/a[2]').click()
        # time.sleep(3)
        BrowserOp.browser_sleep(3)

    def get_total(self):
        return len(self.driver.find_elements(by=By.XPATH, value='//*[@id="pl_feedlist_index"]/div[5]/div/span/ul/li'))

    @staticmethod
    def browser_sleep(sec):
        time.sleep(sec)
