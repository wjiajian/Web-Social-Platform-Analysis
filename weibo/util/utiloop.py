import re
import traceback
from lxml import etree

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
from tqdm import tqdm
import sys


class SaveData:
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

            path1 = '.../video' + str(today) + '/' + str(hour)
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

    # 获取数据
    @staticmethod
    def get_data(div, label, length, bo):
        element = div
        print(length)
        # 用户名
        time.sleep(1)
        try:
            user_name = element.find_element(by=By.XPATH,
                                             value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[1]/div[2]/div[1]/div[2]/a').text
            print("用户名爬取成功")
        except:
            try:
                user_name = element.find_element(by=By.XPATH,
                                                 value=f'//*[@id="pl_feedlist_index"]/div[4]/div[{length}]/div/div[1]/div[2]/div[1]/div[2]/a').text
                print("用户名爬取成功")
            except:
                try:
                    user_name = element.find_element(by=By.XPATH,
                                                     value=f'//*[@id="pl_feedlist_index"]/div[4]/div[{length + 1}]/div/div[1]/div[2]/div[1]/div[2]/a').text
                    print("用户名爬取成功")
                except Exception as e:
                    print("用户名爬取失败")
                    pass

        # 关注列表
        element.find_element(by=By.XPATH,
                             value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[1]/div[2]/div[1]/div[2]/a').click()
        time.sleep(3)
        # 找到元素
        bo.select_new_window()
        time.sleep(1)
        element_new = bo.driver.find_element(by=By.XPATH,
                                             value='//a[@class="ALink_none_1w6rm ProfileHeader_alink_tjHJR ProfileHeader_pointer_2yKGQ"]')
        # 获取href属性值
        href = element_new.get_attribute('href')
        print(href)
        # 提取数字部分
        user_id = href.split('/')[-1].split('?')[0]  # 要爬取的微博用户的user_id列表
        print(user_id)
        fl = Follow()
        try:
            follow_name = fl.start(user_id)
            print("---------------------------------------------")
            print("---------------该用户关注爬取成功---------------")
            print("---------------------------------------------")
            # print(follow_name)
        except:
            pass
        bo.driver.close()
        '''bo.select_new_window()
        bo.driver.find_element(by=By.XPATH,
                               value='//a[2][@class="ALink_none_1w6rm ProfileHeader_alink_tjHJR ProfileHeader_pointer_2yKGQ"]/span/span').click()'''

        # 微博内容
        bo.select_new_window()
        time.sleep(1)
        try:
            content_w = element.find_element(by=By.XPATH,
                                             value=f'//*[@id="pl_feedlist_index"]/div[4]/div[{length}]/div/div[1]/div[2]/p').text
            print("微博内容爬取成功")
            if "展开c" in content_w:
                content_w = element.find_element(by=By.XPATH,
                                                 value=f'//*[@id="pl_feedlist_index"]/div[4]/div[{length}]/div/div[1]/div[2]/p[2]').get_attribute(
                    "innerText")
                print("微博内容爬取成功")
        except:
            try:
                content_w = element.find_element(by=By.XPATH,
                                                 value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[1]/div[2]/p').text
                print("微博内容爬取成功")
                if "展开c" in content_w:
                    content_w = element.find_element(by=By.XPATH,
                                                     value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[1]/div[2]/p[2]').get_attribute(
                        "innerText")
                print("微博内容爬取成功")
            except Exception as e:
                print("微博内容爬取失败")
                pass

        # 获得视频 如果存在
        file_path = ""
        try:
            src = element.find_element(by=By.XPATH, value='.//video').get_attribute("src")
            response = requests.get(src, stream=True)
            path = SaveData.get_path(1)
            file_path = path
            with open(path, "wb") as file:
                file.write(response.content)
        except Exception as e:
            pass

        # 获得图片 如果存在
        try:
            li_list = element.find_elements(by=By.XPATH, value='.//div[@class="media media-piclist"]/ul/li')
            for li in li_list:
                src = li.find_element(by=By.XPATH, value='.//img').get_attribute('src')
                response = requests.get(src, stream=True)
                path = SaveData.get_path(2)
                file_path = str(path) + ","
                with open(path, "wb") as file:
                    file.write(response.content)
        except Exception as e:
            pass
        # 转发数
        try:
            transmitting_num = element.find_element(by=By.XPATH,
                                                    value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[2]/ul/li[1]/a').text
            print("转发数爬取成功")
        except Exception as e:
            print("转发数爬取失败")
            pass
        # 评论数
        try:
            comment_num = element.find_element(by=By.XPATH,
                                               value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[2]/ul/li[2]/a').text
            print("评论数爬取成功")
        except Exception as e:
            print("评论数爬取失败")
            pass
        # 点赞数
        try:
            approval_num = element.find_element(by=By.XPATH,
                                                value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[2]/ul/li[3]/a').text
            print("点赞数爬取成功")
        except Exception as e:
            print("点赞数爬取失败")
            pass
        # 发布时间
        try:
            r_time = element.find_element(by=By.XPATH,
                                          value=f'//*[@id="pl_feedlist_index"]/div[2]/div[{length}]/div/div[1]/div[2]/div[2]/a[1]').text
            print("发布时间爬取成功")
        except Exception as e:
            print("发布时间爬取失败")
            pass
        return (user_name, follow_name, SaveData.clean_emo(content_w), label, SaveData.is_digit(transmitting_num),
                SaveData.is_digit(comment_num), SaveData.is_digit(approval_num), r_time,
                file_path[:-1] if len(file_path) <= 1024 else file_path[:1024])


class BrowserOp:
    def __init__(self, url="https://weibo.com"):
        options = ChromeOptions()
        options.add_experimental_option("detach", True)
        self.driver = Chrome(chrome_options=options)
        self.driver.get(url)
        self.driver.maximize_window()
        time.sleep(20)

    @staticmethod
    def get_config(path="...config/config.json"):
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
        input_f = self.driver.find_element(by=By.XPATH,
                                           value='/html/body/div/div[2]/div[1]/div/div[1]/div/div/'
                                                 'div[1]/div/div[2]/div/span/form/div/input')
        input_f.send_keys(Keys.CONTROL, "a")
        time.sleep(1)
        input_f.send_keys(Keys.BACKSPACE)
        time.sleep(1)
        input_f.send_keys(key)
        time.sleep(1)
        input_f.send_keys(Keys.ENTER)
        time.sleep(1)
        self.select_new_window()
        time.sleep(3)

    # 点击热门按钮
    def click_hot_content(self):
        try:
            self.driver.find_element(by=By.XPATH,
                                     value='//*[@id="pl_feedlist_index"]/div[4]/div[5]/div[1]/h4/a').click()
        except NoSuchElementException:
            try:
                self.driver.find_element(by=By.XPATH,
                                         value='//*[@id="pl_feedlist_index"]/div[2]/div[1]/div[1]/h4/a').click()
            except NoSuchElementException:
                try:
                    self.driver.find_element(by=By.XPATH,
                                             value='//*[@id="pl_feedlist_index"]/div[2]/div[4]/div[1]/h4/a').click()
                except NoSuchElementException:
                    try:
                        self.driver.find_element(by=By.XPATH,
                                                 value='//*[@id="pl_feedlist_index"]/div[4]/div[3]/div[1]/h4/a').click()
                    except NoSuchElementException:
                        # 处理三种情况都不存在的情况
                        print("无法找到任何元素")

        time.sleep(1)
        self.select_new_window()
        time.sleep(2)

    # 点击下一页
    def next_page(self, num):
        self.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
        if num == 2:
            try:
                self.driver.find_element(by=By.XPATH,
                                         value='/html/body/div[1]/div[2]/div/div[2]/div[1]/div[5]/div/a').click()
            except:
                self.driver.find_element(by=By.XPATH,
                                         value='/html/body/div[1]/div[2]/div/div[2]/div[1]/div[3]/div/a').click()

        else:
            try:
                self.driver.find_element(by=By.XPATH,
                                         value='/html/body/div[1]/div[2]/div/div[2]/div[1]/div[5]/div/a[2]').click()
            except:
                self.driver.find_element(by=By.XPATH,
                                         value='/html/body/div[1]/div[2]/div/div[2]/div[1]/div[3]/div/a[2]').click()
        time.sleep(3)

    def get_total(self):
        return len(self.driver.find_elements(by=By.XPATH, value='//*[@id="pl_feedlist_index"]/div[3]/div/span/ul/li'))


class Follow:
    def __init__(self):
        # Follow类初始化
        self.user_id = ''
        self.follow_list = ''  # 存储爬取到的所有关注微博的uri和用户昵称
    
    def deal_html(self, url):
        # 处理html
        try:
            user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36'
            headers = {
                'User_Agent': user_agent,
                'Cookie': '_T_WM=95304686896; WEIBOCN_FROM=1110006030; SCF=AjjHwArbtDh6_oOjPwZNUbDCm7KYdqtkSpZg8h8STSAVPgoNI6-W_BxPJzPl4w0xtBC16NpaAwHjdLq8-dfKG00.; SUB=_2A25JvhY9DeRhGeFG7lAU9CfMyj-IHXVrQLp1rDV6PUNbktAGLWj_kW1NeUD6DmgVkl08sKkXi2H7MhFq9HbUdCA0; SUBP=0033WrSXqPxfM725Ws9jqgMF55529P9D9W5mnMli1.9ycM59OClCr6KO5JpX5KMhUgL.FoMRSKzfSh.7eKe2dJLoIp7LxKML1KBLBKnLxKqL1hnLBoMN1h-ESKB4eh20; SSOLoginState=1689937517; ALF=1692529517; MLOGIN=1; M_WEIBOCN_PARAMS=luicode%3D20000174%26uicode%3D10000011%26fid%3D102803',
                'Connection': 'close'
            }
            html = requests.get(url, headers=headers).content
            selector = etree.HTML(html)
            return selector
        except Exception as e:
            print('Error: ', e)
        traceback.print_exc()

    def get_page_num(self):
        # 获取关注列表页数
        url = "https://weibo.cn/%s/follow" % self.user_id
        selector = self.deal_html(url)
        if selector.xpath("//input[@name='mp']") == []:
            page_num = 1
        else:
            page_num = (int)(
                selector.xpath("//input[@name='mp']")[0].attrib['value'])
        return page_num

    def get_one_page(self, page):
        # 获取第page页的昵称
        print(u'%s第%d页关注%s' % ('-' * 30, page, '-' * 30))
        url = 'https://weibo.cn/%s/follow?page=%d' % (self.user_id, page)
        selector = self.deal_html(url)
        table_list = selector.xpath('//table')
        if (page == 1 and len(table_list) == 0):
            print(u'cookie无效或提供的user_id无效')
        else:
            for t in table_list:
                im = t.xpath('.//a/@href')[-1]
                uri = im.split('uid=')[-1].split('&')[0].split('/')[-1]
                nickname = t.xpath('.//a/text()')[0]
                self.follow_list += nickname + ','
        return self.follow_list

    def get_follow_list(self):
        # 获取关注用户主页地址
        follow_list = ''
        page_num = self.get_page_num()
        if page_num is None:
            page_num = 1
        print(u'用户关注页数：' + str(page_num))
        page1 = 0
        random_pages = random.randint(1, 5)
        for page in tqdm(range(1, page_num + 1), desc=u'关注列表爬取进度'):
            follow_list = self.get_one_page(page)
            # print(follow_list)
            if page - page1 == random_pages and page < page_num:
                time.sleep(random.randint(6, 10))
                page1 = page
                random_pages = random.randint(1, 5)

        print(u'用户关注列表爬取完毕')
        return follow_list

    def initialize_info(self, user_id):
        self.follow_list = ''
        self.user_id = user_id

    def start(self, user_id):
        follow_list = ''
        try:
            self.initialize_info(user_id)
            print('*' * 100)
            follow_list = self.get_follow_list()  # 爬取微博信息
            print(u'信息抓取完毕')
            print('*' * 100)
        except Exception as e:
            print('Error: ', e)
            traceback.print_exc()
        return follow_list
