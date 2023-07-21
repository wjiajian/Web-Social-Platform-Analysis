import time
from selenium.webdriver.common.by import By

from facebook.data.data import MysqlOp
from facebook.util.Exception import TException
from facebook.util.util import BrowserOp, SaveData
from selenium.webdriver.support.ui import WebDriverWait


def get_facebook_data():
    config = BrowserOp.get_config()
    bo = BrowserOp()
    dc = MysqlOp()
    bo.login("email", "password")
    for index, key in enumerate(config["skeys"], 0):
        if index > 0:
            bo.select_new_window(0)
        # 搜索
        bo.search(key)
        results = 0
        pre_content = None
        # 点击帖子
        bo.tiezi()
        WebDriverWait(bo.driver, 10, 0.5).until(lambda driver: driver.find_element(by=By.XPATH,
                                                                                   value='//*[@class="x193iq5w x1xwk8fm"]/div'))
        div_list = bo.driver.find_element(by=By.XPATH,
                                          value='//*[@class="x193iq5w x1xwk8fm"]/div')
        for div in div_list:
            try:
                tuples = SaveData.get_data(div, pre_content, bo.driver)
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
        time.sleep(2)
    # time.sleep(10)


if __name__ == "__main__":
    get_facebook_data()
