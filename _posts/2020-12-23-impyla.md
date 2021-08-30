---
layout: post
title: "使用python连接基于hadoop的数据库"
date: 2020-12-23
description: "使用python连接基于hadoop的数据库。"
tag: Python
---

使用python连接数据库时，频繁手写连接器使人心烦，因此将其包装为一个类，我们需要使用时只需调用它即可。

```python
from time import time
import logging

from impala.dbapi import connect
from decorator import decorator

__all__ = ["Impala", "Hive"]


def stream_logger_factory(func):
    """Factory to create stream logger."""
    assert callable(func) or isinstance(func, str), 'Input must be a function or a function name.'
    if callable(func):
        name = func.__name__.upper()
    else:
        name = func.upper()
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(fmt)
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        logger.addHandler(sh)
    return logger


@decorator
def timer(func, logger=None, *args, **kwargs):
    """Calculate how long the function runs."""
    if not logger:
        logger = stream_logger_factory(func)

    start = int(round(time() * 1000))
    logger.info("Start")
    result = func(*args, **kwargs)
    end = int(round(time() * 1000)) - start
    end /= 1000
    m, s = divmod(end, 60)
    h, m = divmod(m, 60)
    logger.info("Done")
    logger.info("Total execution time: %d:%02d:%02d" % (h, m, s))
    return result


class BaseConnector:
    """Connect and get data from hadoop database."""

    def __init__(self, name):
        """Start a session and create a cursor."""
        self.conn = connect(host=f"{name}.hostserver.com", auth_mechanism="PLAIN",
                            port=YOUR PORT, user="YOUR USERNAME", password="YOUR PASSWORD")
        self.cursor = self.conn.cursor()

    @timer
    def fetch(self, sql, tag=None, to_pandas=True):
        """Run one sql or more and get the result.

        Parameters
        ----------
        sql: str
            * tag=None, one query
            * tag!=None, multiple queries
        tag: str, default None
            Select exactly one query by comment like '/*COMMENT*/' locate at the start and get the result.
        to_pandas: boolean, default True
            Return pandas DataFrame or list.

        Examples
        --------
        >>> s = "describe ods.table; /*I want this one*/select * from ods.table2;"
        >>> with Impala() as db:
        >>>     df = db.fetch(s, tag="I want this one")
        """
        if tag:
            sql_list = [x.strip() for x in self.__split(sql)]
            sql_list = list(filter(lambda x: x.startswith(f"/*{tag}*/"), sql_list))
            assert len(sql_list) == 1, '请检查tag名，重复或不存在'
            sql = sql_list.pop()
        self.cursor.execute(sql)
        if to_pandas:
            from impala.util import as_pandas
            return as_pandas(self.cursor)
        else:
            names = tuple([metadata[0] for metadata in self.cursor.description])
            result = self.cursor.fetchall()
            result.insert(0, names)
            return result

    def _run(self, sql):
        """Run one sql."""
        self.cursor.execute(sql)

    @staticmethod
    def __split(sql):
        sql_list = sql.split(";")
        if sql_list[-1].strip():
            pass
        else:
            sql_list = sql_list[:-1]
        return sql_list

    @timer
    def execute(self, sql, split=True):
        """Run multiple sqls."""
        if split:
            sql = self.__split(sql)
        for x in sql:
            self._run(x)

    def close(self):
        self.conn.close()

    def __call__(self, sql, to_pandas=True):
        """Run multiple sqls and return the last block's result."""
        sql_list = self.__split(sql)
        if sql_list[:-1]:
            self.execute(sql_list[:-1], split=False)
        return self.fetch(sql_list[-1], to_pandas=to_pandas)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class Impala(BaseConnector):

    def __init__(self):
        BaseConnector.__init__(self, "impala")

    @timer
    def upload(self, data, name, schema, dtypes, if_exists="fail"):
        """上传数据至impala数据库

        Parameters
        ----------
        data: pd.DataFrame or List of tuples
            Data.
        name: str
            Name of SQL table.
        schema: str
            Specify the schema.
        dtypes: dict
            Specify each columns' data type.
        if_exists: {"fail", "replace", "append"}, default "fail"
            How to behave if the table already exists.

            * fail: Raise an AssertError.
            * replace: Drop the table before inserting new values.
            * append: Insert new values to the existing table.

        Examples
        --------
        >>> with Impala() as db:
        >>>     db.upload(data, "test_table", "test", dtypes={"dt": "string", "userid": "bigint", "amount": "float"})
        """
        assert if_exists in ("fail", "replace", "append"), "if_exists的值错误"
        if if_exists == "replace":
            self._run(f"drop table if exists {schema}.{name}")
            table = ", ".join([f"{i} {v}" for i, v in dtypes.items()])
            self._run(f"create table {schema}.{name} ({table})")
        else:
            if if_exists == "append":
                assert self.conn.table_exists(name, schema), "该表不存在"
            else:
                assert not self.conn.table_exists(name, schema), "该表已存在"
                table = ", ".join([f"{i} {v}" for i, v in dtypes.items()])
                self._run(f"create table {schema}.{name} ({table})")
        columns = str(tuple(dtypes.keys())).replace("'", "`")
        if isinstance(data, list):
            values = data
        else:
            values = data.apply(lambda x: tuple(x), axis=1).values.tolist()
        values = str(values).strip("[]")
        self._run(f"insert into {schema}.{name} {columns} values {values}")


class Hive(BaseConnector):

    def __init__(self):
        BaseConnector.__init__(self, "hive")
```

使用方法如下所示：

```python
# 例1
with Impala() as db:
    db.execute("**some sql**")
    df = db.fetch("**some sql**")

# 例2
with Impala() as db:
    df = db("**some sql**")

# 例3
db = Impala()
db.fetch("**some sql**")
db.close()
```

