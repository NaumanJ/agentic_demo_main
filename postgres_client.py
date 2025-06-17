import psycopg2
from sqlalchemy import create_engine, inspect
# Class for PostgreSQL functionality
class PostgresClient:
    def __init__(self, connection_string):
        self.engine = create_engine(connection_string)

    def get_schema(self):
        inspector = inspect(self.engine)
        schemas = {}
        for table_name in inspector.get_table_names(schema='public'):
            columns = inspector.get_columns(table_name)
            column_names = [col['name'] for col in columns]
            schemas[table_name] = column_names
        return schemas

    def retrieve_documents(self, question):
        keywords = self._extract_keywords(question)
        retrieved_docs = []
        with self.engine.connect() as conn:
            for keyword in keywords:
                query = f"SELECT * FROM your_table WHERE column_name LIKE '%{keyword}%' LIMIT 5"
                result_set = conn.execute(query)
                rows = result_set.fetchall()
                if rows:
                    retrieved_docs.extend([str(row) for row in rows])
        return retrieved_docs

    def _extract_keywords(self, text):
        # Dummy keyword extraction for illustration
        return text.split()[:5]