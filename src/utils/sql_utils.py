import sqlite3
from typing import Any, List, Tuple
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

class SQLiteDB:
    def __init__(self, db_path: str):
        """Initialize the database connection.

        Args:
            db_path (str): Path to the .db file.
        """
        self.db_path = db_path
        self.connection = None
        self.cursor = None

    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.connection = sqlite3.connect(self.db_path)
            self.cursor = self.connection.cursor()
        except sqlite3.Error as e:
            print(f"Database connection failed: {e}")

    def execute_query(self, raw_query: str) -> List[Tuple[Any, ...]]:
        """Execute a given SQL query and return the results.

        Args:
            raw_query (str): SQL query to be executed.

        Returns:
            List[Tuple[Any, ...]]: The query results.
        """
        # Split the multi_query by semicolons, filtering out empty statements
        queries = [q.strip() for q in raw_query.split(';') if q.strip()]
        sql_results = []

        for query in queries:
            try:
                if not self.cursor:
                    self.connect()
                self.cursor.execute(query)
                if query.lower().startswith("select"):
                    sql_results.append(self.cursor.fetchall())
                else:
                    self.connection.commit()  # Commit changes for insert/update/delete
            except sqlite3.Error as e:
                print(f"Error executing query: {e}")

        return sql_results

    def close(self):
        """Close the database connection."""
        if self.connection:
            self.connection.close()
            self.connection = None
            self.cursor = None

    def output_sql_result(self, query):
        self.connect()
        sql_results = self.execute_query(query)
        self.close()
        return sql_results

def save_to_file(text, index, folder="", easy_read=True):
    file_name = f"retrieved_{index}.txt"
    if easy_read:
        text = text.replace("\\n", "\n")
    folder_path = script_dir + "/../" + folder
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    file_path = folder_path + file_name
    # Open the file in write mode and save the string
    if not os.path.exists(file_path):
        with open(file_path, "x") as file:
            file.write(text)
        print("Saved retrieved data to: " + file_path)
    else:
        last_5_parts = "/".join(file_path.split(os.sep)[-5:])
        # print("Found retrieved data from: " + last_5_parts + " No need to retrieve again")
    return file_path

def save_sql_result(db_path, query, index, folder=""):
    tmp_db = SQLiteDB(db_path=db_path)
    text = tmp_db.output_sql_result(query)
    text_string = str(text)
    file_path = save_to_file(text_string, index, folder)
    return file_path

def truncate_text(text, max_word=1000, index=1):
    """
    Truncate the text to the first max_word words.

    Args:
        text (str): The input text to truncate.
        max_word (int): Maximum number of words allowed in the output.

    Returns:
        str: The truncated text.
    """
    words = text.split()
    split_words = [word for segment in words for word in segment.split(',')]
    if len(split_words) > max_word:
        print("!!!!!!!!!!!!!!!!!!")
        print("WARNING: too much words, truncate to " + str(max_word))
        print("!!!!!!!!!!!!!!!!!!")

        save_to_file(text, index, folder="../tmp/")

        text = ' '.join(split_words[:max_word])
        print("New text after being truncated: ")
        print(text)
        return text
    return text

# Example usage:
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv
    load_dotenv()
    example_db_path = os.getenv("DB_PATH")
    db = SQLiteDB(example_db_path)
    db.connect()

    # example_query = os.getenv("query")
    example_query = "SELECT ADMISSION_LOCATION \n FROM admissions \nWHERE subject_id = 15455707;"

    results = db.execute_query(example_query)
    for row in results:
        print(row)

    # Don't forget to close the database connection when done
    db.close()
