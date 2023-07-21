import time
from selenium.webdriver.common.by import By
from twitter.util.Exception import TException
from twitter.util.utiloop import BrowserOp, SaveData
from twitter.data.data import MysqlOp
from selenium.webdriver.support.ui import WebDriverWait

# "Deep Learning","Computer Vision","Knowledge Graph","Data Mining","AI painting","Autonomous Driving","Speech Recognition",
# "Semiconductor","Integrated circuit","microchip","Cloud Computing","Internet of Things","Decentralized computing",
# "Edge Computing","Brain Computer Interface","Natural Language Processing","OpenAI"

# 网速快可调整等待时间


def get_twitter_data():
    config = BrowserOp.get_config()
    bo = BrowserOp()
    bo.login("account", "password")
    num = 0
    flag = False
    dc = MysqlOp()
    for index, key in enumerate(config["skeys"], 0):
        bo.click_explore()
        bo.search(key)
        results = []
        pre_content = None
        while True:
            # 获取到所有内容元素
            WebDriverWait(bo.driver, 10, 0.5).until(lambda driver: driver.find_element(by=By.XPATH,
                                                                                       value='//*[@id="react-root"]/div/div/div[2]/main/div/div/div/'
                                                                                             'div[1]/div/div[3]/section/div/div/div/div/div'))
            div_list = bo.driver.find_elements(by=By.XPATH,
                                               value='//div[@data-testid="cellInnerDiv"]')
            # pre_content：上一次获得的推文内容
            # print(div_list)
            length = 0
            for div in div_list:
                print(div)
                try:
                    tuples = SaveData.get_data(div, pre_content, bo.driver, length)
                    # 结束判断
                    if tuples == TException.NO_NEW_CONTENT:
                        continue
                    elif tuples == TException.STRUCTURE_EXCEPTION:
                        continue
                    pre_content = tuples[0]
                    if tuples[1]:
                        results.append(tuples[1])
                    if (num := num + 1) % config["batch_size"] == 0:
                        dc.batch_insert(results)
                        results.clear()
                except Exception as e:
                    print(e)
                length = length + 1
            print("爬取了", num, "条数据")
            # if num % 550 == 0:
            #     bo.driver.refresh()
            #     # time.sleep(10)
            #     BrowserOp.browser_sleep(10)
            # else:
            check_heights = bo.driver.execute_script("return document.documentElement.scrollTop || "
                                                     "window.pageYOffset || document.body.scrollTop;")
            bo.driver.execute_script("window.scrollTo(0,document.body.scrollHeight)")
            BrowserOp.browser_sleep(5)
            check_heighte = bo.driver.execute_script("return document.documentElement.scrollTop || "
                                                     "window.pageYOffset || document.body.scrollTop;")
            if check_heights == check_heighte:
                if len(results) != 0:
                    dc.batch_insert(results)
                    results.clear()
                break
            # time.sleep(7)

            if flag:
                flag = False
                print("外层for循环结束")
                break
    print("程序结束")


if __name__ == "__main__":
    get_twitter_data()
