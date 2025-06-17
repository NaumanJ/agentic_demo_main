# postgres_manager.py

import os
import glob
import re
import pandas as pd
from sqlalchemy import create_engine
import logging

class PostgresManager:
    def __init__(self, user, password, host, port, database):
        self.connection_string = f'postgresql://{user}:{password}@{host}:{port}/{database}'
        self.engine = create_engine(self.connection_string)
        self.connection = self.engine.connect()

    def load_csv_files(self, directory):
        csv_files = glob.glob(os.path.join(directory, "**/*.csv"), recursive=True)
        for csv_file in csv_files:
            table_name = os.path.basename(csv_file)[:-4]
            sanitized_table_name = self.sanitize_table_name(table_name)
            self.load_data_into_postgres(csv_file, sanitized_table_name)

    def load_data_into_postgres(self, csv_file, table_name):
        """Load a CSV file into a PostgreSQL table using Pandas."""
        try:
            df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        except UnicodeDecodeError:
            df = pd.read_csv(csv_file, encoding='latin1', on_bad_lines='skip')
        except Exception as e:
            logging.error(f"Error reading CSV with Pandas: {e}")
            return

        # Load DataFrame into PostgreSQL
        try:
            df.to_sql(table_name, self.engine, if_exists='fail', index=False)
            logging.info(f"Loaded data into table {table_name}")
        except ValueError as e:
            logging.info(f"Table already exists in PostgreSQL: {e}")
        except Exception as e:
            logging.error(f"Error loading data into PostgreSQL: {e}")

    def close(self):
        self.connection.close()

    @staticmethod
    def sanitize_table_name(name):
        """Sanitize the table name by replacing unsafe characters."""
        # Replace spaces and hyphens with underscores
        name = re.sub(r'[\s\-]+', '_', name)
        # Remove any characters that are not alphanumeric or underscore
        name = re.sub(r'[^\w]', '', name)
        # Ensure the name doesn't start with a number
        if re.match(r'^\d', name):
            name = f"t_{name}"
        return name