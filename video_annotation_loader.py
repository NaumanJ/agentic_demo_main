import os
import pandas as pd
import logging
from sqlalchemy import create_engine, text
from postgres_manager import PostgresManager
from postgres_client import PostgresClient

db_config = {
    'dbname': 'postgres',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': '5433'
}

class AnnotationLoader:
    def __init__(self, db_config):
        """
        Initialize the AnnotationLoader with the PostgreSQL configuration.
        """
        self.manager = PostgresManager(
            db_config['user'],
            db_config['password'],
            db_config['host'],
            db_config['port'],
            db_config['dbname']
        )
        self.client = PostgresClient(self.manager.connection_string)
        # Establish the connection and create a cursor
        self.connection = self.manager.engine.raw_connection()
        self.cursor = self.connection.cursor()

    def drop_annotation_table(self, table_name="video_annotations"):
        """Drop the table if it exists."""
        drop_table_query = f"DROP TABLE IF EXISTS {table_name}"
        try:
            # Using begin() to handle transaction scope automatically
            with self.manager.engine.begin() as conn:
                conn.execute(text(drop_table_query))
            logging.info(f"Table '{table_name}' dropped.")
        except Exception as e:
            logging.error(f"Error dropping Table '{table_name}': {e}")

    def create_annotation_table(self, table_name="video_annotations"):
        """Create the video annotations table if it doesn't exist."""
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            video_file_name VARCHAR(255) NOT NULL,
            start_duration VARCHAR(10) NOT NULL,
            end_duration VARCHAR(10) NOT NULL,
            annotation TEXT NOT NULL,
            annotation_tsv TSVECTOR
        );
        """
        try:
            # Using begin() to handle transaction scope automatically
            with self.manager.engine.begin() as conn:
                conn.execute(text(create_table_query))
            logging.info(f"Table '{table_name}' created.")
        except Exception as e:
            logging.error(f"Error creating Table '{table_name}': {e}")
            return  # Exit if table creation fails
        
        query_ddl = f"""SELECT *
                        FROM information_schema.columns
                        WHERE table_schema = 'public'
                        AND table_name   = '{table_name}'
                            ;"""
        try:
            self.cursor.execute(query_ddl)
            logging.info(f"Retrieving Table '{table_name}' DDL.")
            print(self.cursor.fetchall())
        except Exception as e:
            logging.error(f"Error creating Table '{table_name}': {e}")
            return  # Exit if table creation fails
        # Add GIN index for full-text search if it does not exist
        create_text_index_sql = f"""
        CREATE INDEX IF NOT EXISTS annotation_tsv_idx ON {table_name} USING GIN (annotation_tsv);
        """
        try:
            # Using begin() to handle transaction scope automatically
            with self.manager.engine.begin() as conn:
                conn.execute(text(create_text_index_sql))
            logging.info(f"Full-text search index 'annotation_tsv_idx' created.")
        except Exception as e:
            logging.error(f"Error creating index on '{table_name}': {e}")

    def insert_annotation(self, video_file_name, start_duration, end_duration, annotation, table_name="video_annotations"):
        """
        Insert a new annotation into the database and update the TSVECTOR field.
        """
        try:
            query = f"""
            INSERT INTO {table_name} (video_file_name, start_duration, end_duration, annotation, annotation_tsv)
            VALUES (%s, %s, %s, %s, to_tsvector(%s))
            """
            self.cursor.execute(query, (video_file_name, start_duration, end_duration, annotation, annotation))
            self.connection.commit()  # Commit after successful insertion
        except Exception as e:
            self.connection.rollback()  # Rollback the transaction on error
            logging.error(f"Error inserting data into '{table_name}': {e}")

    def load_annotations_from_csv(self, csv_file_path):
        """
        Load annotations from a CSV file into the PostgreSQL table.

        Args:
            csv_file_path (str): Path to the CSV file.
        """
        df = pd.DataFrame()
        try:
            # Read CSV data into a DataFrame
            df = pd.read_csv(csv_file_path, encoding='latin1', header=0)
        except UnicodeDecodeError:
            # Try another encoding if the first fails
            df = pd.read_csv(csv_file_path, encoding='cp1252', header=0)

        if df.empty:
            logging.warning(f"No data found in {csv_file_path}. Skipping.")
            return

        # Adjust the column names
        df.columns = ['video_file_name', 'time_range', 'annotation']
        df[['start_duration', 'end_duration']] = df['time_range'].str.split(' -> ', expand=True)
        df.drop(columns=['time_range'], inplace=True)

        # Insert data into the table
        try:
            for index, row in df.iterrows():
                video_file_name = row['video_file_name']
                start_duration = row['start_duration']
                end_duration = row['end_duration']
                annotation = row['annotation']
                self.insert_annotation(video_file_name, start_duration, end_duration, annotation)
            logging.info(f"Loaded data from {csv_file_path} into 'video_annotations'.")
        except Exception as e:
            logging.error(f"Error loading data from {csv_file_path}: {e}")

    def global_search_by_annotation(self, search_text):
        """
        Perform a full-text search on the annotations and return video details.

        Args:
            search_text (str): The text to search for.

        Returns:
            list of tuples: Each tuple contains (video_file_name, start_duration, end_duration, annotation).
        """
        query = """
        SELECT video_file_name, start_duration, end_duration, annotation
        FROM video_annotations
        WHERE annotation_tsv @@ to_tsquery(%s)
        """
        ts_query = " & ".join(search_text.split())
        self.cursor.execute(query, (ts_query,))
        return self.cursor.fetchall()

    def search_video_by_annotation(self, selected_video, search_text):
        """
        Search for video details for a specific video using a full-text search on the annotations.

        Args:
            selected_video (str): The video file name to search within.
            search_text (str): The text to search for.

        Returns:
            list of tuples: Each tuple contains (video_file_name, start_duration, end_duration, annotation).
        """
        query = """
        SELECT video_file_name, start_duration, end_duration, annotation
        FROM video_annotations
        WHERE video_file_name = %s
        AND annotation_tsv @@ to_tsquery(%s)
        """
        ts_query = " & ".join(search_text.split())
        self.cursor.execute(query, (selected_video, ts_query))
        return self.cursor.fetchall()

    def search_video_by_duration(self, selected_video, time):
        """
        Search for video details where the given time falls within the start and end duration range.

        Args:
            selected_video (str): The video file name to search within.
            time (str): The time in 'mm:ss' format to search for.

        Returns:
            list of tuples: Each tuple contains (video_file_name, start_duration, end_duration, annotation).
        """
        query = """
        SELECT video_file_name, start_duration, end_duration, annotation
        FROM video_annotations
        WHERE video_file_name = %s
        AND %s >= start_duration AND %s <= end_duration
        """
        self.cursor.execute(query, (selected_video, time, time))
        return self.cursor.fetchall()

    def close(self):
        """Close the connection and cursor."""
        self.cursor.close()
        self.connection.close()
        self.manager.close()