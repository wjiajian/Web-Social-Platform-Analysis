import pymysql


class MysqlOp:
    def __init__(self, host='localhost', user='root', passwd='root', port=3307, autocommit=True):
        self.session = pymysql.connect(host='localhost', user='root', passwd='password', port=3306)
        self.session.autocommit(autocommit)
        self.cursor = self.session.cursor()

    def batch_insert(self, data):
        sql = "INSERT INTO `test`.`facebook_t`(`user_name`, `content`,`comment_num`,`transmitting_num`,`approval_num`,`view_num`) VALUES(%s, %s, %s, %s, %s, %s)"
        self.cursor.executemany(sql, data)