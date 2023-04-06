from tqdm import tqdm
import xlrd
import pymysql
from config import faq_runner_config
DATA_BASE = "input/data/faq/"

def getConn():
    args = faq_runner_config.get("conn")
    conn = pymysql.connect(**args)
    return conn

def excel2mysql(excelName, drop=True):
    #下面代码作用：获取到excel中的字段和数据
    excel = xlrd.open_workbook(excelName)
    sheet = excel.sheet_by_index(0)
    row_number = sheet.nrows
    data_list = []
    for i in range(1,row_number):
        data_list.append(sheet.row_values(i))

    conn = getConn()
    table = faq_runner_config.get("table_name")
    cursor = conn.cursor()
    if drop:
        print(f"Dropping existing {table}...")
        drop_sql = "drop table if exists {}".format(table)
        cursor.execute(drop_sql)
    print(f"Creating {table}...")
    create_sql = f"CREATE TABLE IF NOT EXISTS `{table}`( \
                `id` BIGINT UNSIGNED AUTO_INCREMENT NOT NULL PRIMARY KEY, \
                `question` VARCHAR(1024) NOT NULL, \
                `answer` VARCHAR(4096) NOT NULL) \
                ENGINE=InnoDB default charset=utf8mb4;"
    cursor.execute(create_sql)
    print(f"Inserting data...")
    for _id, data in tqdm(enumerate(data_list)):
        new_data = ["'{}'".format(i) for i in data]
        insert_sql = "insert into {} values({})".format(table, ','.join([str(_id + 1)] + new_data))
        cursor.execute(insert_sql)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    excel2mysql(f"{DATA_BASE}/file/qa100.xls")