import pandas as pd
import pymysql
from request import *


db_name = "maaDB"


# def dbconnect():
#   try:
#      db = pymysql.connect(
#         host="database-1.cqnkqcoyeu6x.eu-west-2.rds.amazonaws.com",
#        user="admin",
#       password="123456789",
#  )
# except Exception as e:
#    print("Can't connect to database")
# return db

# db = pymysql.connect(
#           host="database-1.cqnkqcoyeu6x.eu-west-2.rds.amazonaws.com",
#          user="admin",
#         password="123456789",
#    )

# RETURN TYPE
def dbconnect():
    try:
        db = pymysql.connect(
            host="database-1.cqnkqcoyeu6x.eu-west-2.rds.amazonaws.com",
            user="admin",
            password="123456789",
        )
    except Exception as e:
        print("Can't connect to database")
    return db


class Table:
    db_name = "maaDB"
    db = pymysql.connect(
        host="database-1.cqnkqcoyeu6x.eu-west-2.rds.amazonaws.com",
        user="admin",
        password="123456789",
    )

    def dbconnect():
        try:
            db = pymysql.connect(
                host="database-1.cqnkqcoyeu6x.eu-west-2.rds.amazonaws.com",
                user="admin",
                password="123456789",
            )
        except Exception as e:
            print("Can't connect to database")
        return db

    def __init__(self, table_name: str, columns: dict):
        self.table_name = table_name
        self.columns = columns

    def create_table(self) -> None:
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE {db_name}".format(db_name=db_name))
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {db_name}")
            sql = """CREATE TABLE {table_name} (
                 ID int NOT NULL AUTO_INCREMENT""".format(
                table_name=self.table_name
            )
            for key in self.columns:
                sql = sql + ", " + str(key) + " " + str(self.columns.get(key))
            sql = sql + ",PRIMARY KEY (ID) )"
            cursor.execute(sql)
            db.commit()
            cursor.close()
            print("Table Created")

        except Exception as e:
            print(e)

    @staticmethod
    def check_tables() -> tuple:
        db = dbconnect()
        cursor = db.cursor()
        cursor.execute("USE {db_name}".format(db_name=db_name))
        sql = "SHOW TABLES"
        cursor.execute(sql)
        return cursor.fetchall()

    def show_table_entries(self) -> None:
        db = dbconnect()
        cursor = db.cursor()
        cursor.execute("USE {db_name}".format(db_name=db_name))
        sql = "SELECT * FROM {table_name}".format(table_name=self.table_name)
        cursor.execute(sql)
        result = cursor.fetchall()
        print("Total rows are: ", len(result))
        # print(result)
        cursor.close()
        return result

    # COUNT NUMBER OF ENTIRES IN TABLE
    def count_table_index(self) -> int:
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE {db_name}".format(db_name=db_name))
            sql = "SELECT COUNT ('ID') FROM {table_name}".format(
                table_name=self.table_name
            )
            cursor.execute(sql)
            count = cursor.fetchall()
            return int(count)

        except Exception as e:
            print(e)

    # DELETE ENTRY FROM TABLE
    # MAYBE PUT CONDITION INSTEAD OF ID LIKE WHERE CustomerName="Alfreds Futterkiste"
    def delete_from_table(self, condition: str) -> None:
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE maaDB")
            sql = "DELETE FROM {table_name} WHERE {condition}".format(
                table_name=self.table_name, condition=condition
            )
            cursor.execute(sql)
            db.commit()
            cursor.close()
            print("Deletion completed!")
        except Exception as e:
            print(e)

    def add_columns(self, columns: dict) -> None:  # CHANGE self.columns with append
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE {db_name}".format(db_name=db_name))
            sql = "ALTER TABLE {table_name}".format(table_name=self.table_name)
            for key in columns:
                sql = (
                    sql + " ADD COLUMN " + str(key) + " " + str(columns.get(key)) + ","
                )
            cursor.execute(sql)
            cursor.close()
            db.commit()
            self.columns = self.columns.update(columns)
            print("Columns added successfully")

        except Exception as e:
            print(e)

    def insert_db(self, columns: dict, values: list) -> None:
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE {db_name}".format(db_name=db_name))
            if len(columns) == len(values):
                sql = "INSERT INTO {table_name} (".format(table_name=self.table_name)
                for key in columns:
                    sql = sql + str(key) + ", "
                sql = sql[:-2] + ") VALUES ("
                for value in values:
                    sql = sql + '"' + str(value) + '"' + ", "
                sql = sql[:-2] + ")"
                cursor.execute(sql)
                cursor.close()
                db.commit()
                print("Insertion completed.")
            else:
                print("Number of columns != number of values!")
        except Exception as e:
            print(e)

    def drop_table(self) -> None:
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE {db_name}".format(db_name=db_name))
            sql = "DROP TABLE {table_name}".format(table_name=self.table_name)
            cursor.execute(sql)
            cursor.close()
            db.commit()
            print(f"Table {self.table_name} dropped successfully!")
        except Exception as e:
            print(e)

    def table_to_pddf(self, columns_few: str) -> None:
        try:
            db = dbconnect()
            cursor = db.cursor()
            cursor.execute("USE {db_name}".format(db_name=db_name))
            sql = "SELECT {columns} FROM {table_name}".format(
                columns=columns_few, table_name=self.table_name
            )
            sql_query = pd.read_sql_query(sql, db)
            cursor.close()
            return pd.DataFrame(sql_query)
        except Exception as e:
            print(e)
