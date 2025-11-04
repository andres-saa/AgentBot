
import os
import inspect
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor, Json
from pydantic import BaseModel

class DBModel(BaseModel):
    model_config = {"extra": "ignore"}
    __schema__: str = ''
    __tablename__: str | None = None

    @classmethod
    def _to_snake(cls, name: str) -> str:
        return re.sub(r'(?<!^)(?=[A-Z])', '_', name).lower()

    @classmethod
    def table_fullname(cls) -> str:
        name = cls.__tablename__ or cls._to_snake(cls.__name__)
        return f'{cls.__schema__}.{name}' if cls.__schema__ else name

load_dotenv()
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

class Db:
    def __init__(self) -> None:
        self.conn_str = (
            f"dbname={DB_NAME} user={DB_USER} password={DB_PASSWORD} "
            f"host={DB_HOST} port={DB_PORT}"
        )
        self.conn = psycopg2.connect(self.conn_str)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_connection()

    def close_connection(self):
        self.conn.close()

    @staticmethod
    def _get_table(model_or_cls: Union[DBModel, type[DBModel]]) -> str:
        cls = model_or_cls if isinstance(model_or_cls, type) else model_or_cls.__class__
        if hasattr(cls, "table_fullname"):
            return cls.table_fullname()
        return re.sub(r'(?<!^)(?=[A-Z])', '_', cls.__name__).lower()

    @staticmethod
    def _to_payload(data: BaseModel) -> Dict[str, Any]:
        return data.model_dump(exclude_none=True)

    def execute_query(
        self,
        query: str,
        params: Optional[Union[Dict[str, Any], Tuple, List]] = None,
        fetch: bool = False,
    ):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                result = None
                if fetch:
                    rows = cursor.fetchall()
                    if not rows:
                        result = None
                    elif len(rows) == 1:
                        result = rows[0]
                    else:
                        result = rows
                self.conn.commit()
                return result
        except Exception as e:
            self.conn.rollback()
            print(f"An error occurred: {e}")

    def _process_json_params(self, params):
        if params is None:
            return None
        if isinstance(params, (list, tuple)):
            return type(params)(
                Json(p) if isinstance(p, (dict, list)) else p for p in params
            )
        if isinstance(params, dict):
            return {k: Json(v) if isinstance(v, (dict, list)) else v for k, v in params.items()}
        return params

    def execute_query_json(
        self,
        query: str,
        params: Optional[Union[Dict, Tuple, List]] = None,
        fetch: bool = False,
    ):
        processed = self._process_json_params(params)
        return self.execute_query(query, processed, fetch)

    def build_select_query(
        self,
        target: Union[type[DBModel], DBModel, str],
        fields: Optional[List[str]] = None,
        condition: str = '',
        order_by: str = '',
        limit: int = 0,
        offset: int = 0,
        *,
        schema: str = ''
    ) -> str:
        if isinstance(target, str):
            table = f'{schema}.{target}' if schema else target
        else:
            table = self._get_table(target)
        cols = ', '.join(fields) if fields else '*'
        query = f'SELECT {cols} FROM {table}'
        if condition:
            query += f' WHERE {condition}'
        if order_by:
            query += f' ORDER BY {order_by}'
        if limit:
            query += f' LIMIT {limit}'
        if offset:
            query += f' OFFSET {offset}'
        return query

    def build_insert_query(
        self,
        data: DBModel,
        returning: str = ''
    ) -> Tuple[str, Dict[str, Any]]:
        table = self._get_table(data)
        payload = self._to_payload(data)
        cols = ', '.join(payload.keys())
        vals = ', '.join(f'%({k})s' for k in payload)
        query = f'INSERT INTO {table} ({cols}) VALUES ({vals})'
        if returning:
            query += f' RETURNING {returning}'
        return query, payload

    def build_bulk_insert_query(
        self,
        data_list: List[DBModel],
        returning: str = ''
    ) -> Tuple[str, List[Dict[str, Any]]]:
        if not data_list:
            raise ValueError("data_list no puede estar vacío")
        table = self._get_table(data_list[0])
        first_payload = self._to_payload(data_list[0])
        cols = ', '.join(first_payload.keys())
        placeholders = ', '.join(f'%({k})s' for k in first_payload)
        values_block = ', '.join(f'({placeholders})' for _ in data_list)
        query = f'INSERT INTO {table} ({cols}) VALUES {values_block}'
        if returning:
            query += f' RETURNING {returning}'
        params = [self._to_payload(m) for m in data_list]
        return query, params

    def execute_bulk_insert(
        self,
        query: str,
        params: List[Dict[str, Any]],
        fetch: bool = False,
    ):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.executemany(query, params)
                self.conn.commit()
                if fetch:
                    return cursor.fetchall()
        except Exception as e:
            self.conn.rollback()
            print(f"An error occurred: {e}")

    def build_update_query(
        self,
        data: DBModel,
        condition: str,
        returning: str = ''
    ) -> Tuple[str, Dict[str, Any]]:
        table = self._get_table(data)
        payload = self._to_payload(data)
        set_clause = ', '.join(f'{k} = %({k})s' for k in payload)
        query = f'UPDATE {table} SET {set_clause} WHERE {condition}'
        if returning:
            query += f' RETURNING {returning}'
        return query, payload

    def execute_bulk_update(
        self,
        query: str,
        params: List[Dict[str, Any]],
        fetch: bool = False,
    ):
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.executemany(query, params)
                self.conn.commit()
                if fetch:
                    return cursor.fetchall()
        except Exception as e:
            self.conn.rollback()
            print(f"An error occurred: {e}")

    def build_soft_delete_query(
        self,
        model_cls: type[DBModel],
        condition: str,
        returning: str = ''
    ) -> str:
        table = self._get_table(model_cls)
        query = f'UPDATE {table} SET exist = FALSE WHERE {condition}'
        if returning:
            query += f' RETURNING {returning}'
        return query

    def build_delete_query(
        self,
        model_cls: type[DBModel],
        condition: str,
        returning: str = ''
    ) -> str:
        table = self._get_table(model_cls)
        query = f'DELETE FROM {table} WHERE {condition}'
        if returning:
            query += f' RETURNING {returning}'
        return query

    def fetch_one(self, query: str, params=None):
        return self.execute_query(query, params, fetch=True)

    def fetch_all(self, query: str, params=None):
        result = self.execute_query(query, params, fetch=True)
        return result

    def cargar_archivo_sql(self, nombre_archivo: str) -> Optional[str]:
        try:
            ruta_llamador = os.path.dirname(
                os.path.abspath(inspect.stack()[1].filename)
            )
            ruta_archivo = os.path.join(ruta_llamador, nombre_archivo)
            with open(ruta_archivo, "r", encoding="utf-8") as archivo:
                return archivo.read()
        except FileNotFoundError:
            print(f"El archivo '{nombre_archivo}' no fue encontrado en '{ruta_llamador}'.")
        except Exception as e:
            print(f"Ocurrió un error al leer el archivo: {e}")
        return None
