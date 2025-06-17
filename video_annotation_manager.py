import psycopg2
from psycopg2 import sql

class AnnotationManager:
    def __init__(self, db_config):
        """
        Initialize the AnnotationManager with the database connection.

        Args:
            db_config (dict): A dictionary with keys 'dbname', 'user', 'password', 'host', 'port'.
        """
        self.connection = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        self.cursor = self.connection.cursor()
        self._initialize_database()

    def _initialize_database(self):
        """Initialize the database schema and index for full-text search."""
        # Create table if it does not exist
        self.cursor.execute("""
        CREATE TABLE IF NOT EXISTS video_annotations (
            id SERIAL PRIMARY KEY,
            video_file_name VARCHAR(255) NOT NULL,
            start_duration VARCHAR(10) NOT NULL,
            end_duration VARCHAR(10) NOT NULL,
            annotation TEXT NOT NULL,
            annotation_tsv TSVECTOR
        );
        """)
        self.connection.commit()

        # Add GIN index for full-text search if it does not exist
        self.cursor.execute("""
        CREATE INDEX IF NOT EXISTS annotation_tsv_idx ON video_annotations USING GIN (annotation_tsv);
        """)
        self.connection.commit()

    def insert_annotation(self, video_file_name, start_duration, end_duration, annotation):
        """
        Insert a new annotation into the database and update the TSVECTOR field.

        Args:
            video_file_name (str): The name of the video file.
            start_duration (str): The start duration of the annotation.
            end_duration (str): The end duration of the annotation.
            annotation (str): The text annotation.
        """
        query = """
        INSERT INTO video_annotations (video_file_name, start_duration, end_duration, annotation, annotation_tsv)
        VALUES (%s, %s, %s, %s, to_tsvector(%s))
        """
        self.cursor.execute(query, (video_file_name, start_duration, end_duration, annotation, annotation))
        self.connection.commit()

    def load_annotations_from_csv(self, csv_file_path):
        """
        Load annotations from a CSV file into the database.

        Args:
            csv_file_path (str): The path to the CSV file containing the annotations.
        """
        import csv

        with open(csv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # Skip header row if there is one
            for row in reader:
                video_file_name, time_range, annotation = row
                start_duration, end_duration = time_range.split(" -> ")
                self.insert_annotation(video_file_name, start_duration, end_duration, annotation)

    def search_by_annotation(self, search_text):
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
        LIMIT 1
        """
        # Convert the search text into a tsquery format, e.g., 'word1 & word2'
        ts_query = " & ".join(search_text.split())
        self.cursor.execute(query, (ts_query,))
        return self.cursor.fetchall()

    def close_connection(self):
        """Close the database connection."""
        self.cursor.close()
        self.connection.close()

# Example usage
if __name__ == "__main__":
    # Database configuration
    db_config = {
        'dbname': 'postgres',
        'user': 'postgres',
        'password': 'password',
        'host': 'localhost',
        'port': '5433'
    }

    # Initialize the AnnotationManager
    manager = AnnotationManager(db_config)

    # Load annotations from a CSV file
    manager.load_annotations_from_csv("All_2022WC_Spain_Other_FIFA_FOX.csv")
    manager.load_annotations_from_csv("All_Ronaldo_IG.csv")

    # Perform a full-text search
    results = manager.search_by_annotation("man kicking ball")
    for result in results:
        print("Video:", result[0])
        print("Start:", result[1])
        print("End:", result[2])
        print("Annotation:", result[3])

    # Close the connection
    manager.close_connection()