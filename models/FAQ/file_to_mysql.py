import xlrd
import pymysql

def getConn(database='qa100'):
    args = dict(
        host='localhost',
        user='faq',
        passwd='123456',
        db=database,
        charset='utf8'
    )
    conn = pymysql.connect(**args)
    return conn

def excel2mysql(excelName,database='qa100',table='qa100'):
    #下面代码作用：获取到excel中的字段和数据
    excel = xlrd.open_workbook(excelName)
    sheet = excel.sheet_by_index(0)
    row_number = sheet.nrows
    column_number = sheet.ncols
    field_list = sheet.row_values(0)
    data_list = []
    for i in range(1,row_number):
        data_list.append(sheet.row_values(i))
    #下面代码作用：根据字段创建表，根据数据执行插入语句
    conn = getConn(database)
    cursor = conn.cursor()
    drop_sql = "drop table if exists {}".format(table)
    cursor.execute(drop_sql)
    create_sql = "create table {}(".format(table)
    for field in field_list[:-1]:
        create_sql += "{} varchar(2048),".format(field)
    create_sql += "{} varchar(2048))".format(field_list[-1])
    cursor.execute(create_sql)
    for data in data_list:
        new_data = ["'{}'".format(i) for i in data]
        insert_sql = "insert into {} values({})".format(\
            table,','.join(new_data))
        cursor.execute(insert_sql)
    conn.commit()
    conn.close()

if __name__ == '__main__':
    excel2mysql("/home/gyc/bert-kbqa/models/FAQ/data/file/qa100.xls")