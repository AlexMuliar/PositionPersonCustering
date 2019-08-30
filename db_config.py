import pymysql.cursors

def get_conn():
    connection = pymysql.connect(host='localhost',
                                user='molengo',
                                password='password',
                                db='molengo',
                                charset='utf8mb4',
                                cursorclass=pymysql.cursors.DictCursor)
    return connection

