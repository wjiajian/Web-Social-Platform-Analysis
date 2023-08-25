import pymysql


# 打开数据库连接
def get_session():
    try:
        db = pymysql.connect(host='localhost', user='root', passwd='jiajian233', port=3306)
        db.autocommit(True)
        print('连接成功！')
    except ValueError:
        print('something wrong!')
    return db


def get_cursor(session):
    return session.cursor()


def batch_insert(cursor, data):
    sql = "INSERT INTO `test`.`weibo`(`user_name`, `content_w`,`transmitting_num`,`comment_num`,`approval_num`,`time`,`file_path`) VALUES(%s, %s, %s, %s, %s, %s, %s)"
    cursor.executemany(sql, data)


class MysqlOp:
    def __init__(self,host='localhost', user='root', passwd='jiajian233', port=3306,autocommit=True):
        self.db = pymysql.connect(host='localhost', user='root', passwd='jiajian233', port=3306)
        self.db.autocommit(autocommit)
        self.cursor = self.db.cursor()


    def batch_insert(self, data):
        sql = "INSERT INTO `test`.`weibo_new_new`(`user_name`,`guanzhu`,`content_w`,`label`,`transmitting_num`,`comment_num`,`approval_num`,`time`,`file_path`) VALUES(%s, %s, %s, %s, %s, %s, %s, %s, %s)"
        self.cursor.executemany(sql, data)
