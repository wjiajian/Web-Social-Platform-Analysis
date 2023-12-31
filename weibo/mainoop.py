import time

from selenium.webdriver.common.by import By
from data import MysqlOp as MysqlOp
from weibo.util.utiloop import SaveData, BrowserOp

# 深度学习没爬,计算机视觉没爬完
# 爬了的：ChatGPT,AI画图,深度学习

# "深度学习","ChatGPT","计算机视觉","知识图谱","数据挖掘",
# "AI画图","自动驾驶","语音识别","算力","半导体",
# "国产芯片","元宇宙","云计算","物联网","加密技术",
# "数字化","去中心化","5G","电子支付","区块链",
# "边缘计算","脑机接口","自然语言处理","OpenAI"


def get():
    config = BrowserOp.get_config()
    # root_urls = 'https://weibo.com'
    # 操作浏览器对象
    bo = BrowserOp()
    # 操作数据库对象
    dc = MysqlOp()
    i = 0
    for index, key in enumerate(config["skeys"], 0):
        if index > 0:
            bo.select_new_window(0)
        # 搜索关键字
        bo.search(key)
        # 点击热门
        bo.click_hot_content()
        # 获取搜索内容的总页数，最多50页
        total = bo.get_total()
        results = []
        num = 1
        while True:
            # 获取当页所有包含微博内容div
            # //*[@id="pl_feedlist_index"]/div[2]/div     不是话题
            # //*[@id="pl_feedlist_index"]/div[4]/div     话题
            print('第%d页微博' %num)
            div_list = bo.driver.find_elements(by=By.XPATH, value='//*[@id="pl_feedlist_index"]/div[2]/div')
            print(div_list)
            length = 1
            for div in div_list:
                results.append(SaveData.get_data(div, key, length, bo))
                # print('----------------------------------')
                # print(results)
                # print('----------------------------------')
                length = length + 1
                if (i := i + 1) % config["batch_size"] == 0:
                    dc.batch_insert(results)
                    results.clear()
            if (num := num + 1) > total:
                break
            time.sleep(2)
            bo.next_page(num)
    print("总共爬取", i, "条数据")
    time.sleep(10)


if __name__ == "__main__":
    get()
