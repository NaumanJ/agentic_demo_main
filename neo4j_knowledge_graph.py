# Neo4j Knowledge Graph Class
from neo4j import GraphDatabase
class Neo4jKnowledgeGraph:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    def close(self):
        self.driver.close()

    def create_document_node(self, title, content):
        with self.driver.session() as session:
            session.write_transaction(self._create_document_node, title, content)

    @staticmethod
    def _create_document_node(tx, title, content):
        tx.run("MERGE (d:Document {title: $title}) SET d.content = $content",
               title=title, content=content)

    def create_relationship(self, title1, title2, relationship):
        with self.driver.session() as session:
            session.write_transaction(self._create_relationship, title1, title2, relationship)

    @staticmethod
    def _create_relationship(tx, title1, title2, relationship):
        tx.run("MATCH (d1:Document {title: $title1}), (d2:Document {title: $title2}) "
               "MERGE (d1)-[:RELATED_TO {type: $relationship}]->(d2)",
               title1=title1, title2=title2, relationship=relationship)

    def get_top_keywords(self, limit=5):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_top_keywords, limit)
        return result

    @staticmethod
    def _get_top_keywords(tx, limit):
        query = """
        MATCH (d:Document)
        WITH split(d.content, ' ') AS words
        UNWIND words AS word
        WITH word, count(*) AS frequency
        WHERE size(word) > 3
        RETURN word, frequency
        ORDER BY frequency DESC
        LIMIT $limit
        """
        result = tx.run(query, limit=limit)
        keywords = [record["word"] for record in result]
        return keywords

    def search_documents(self, keywords):
        with self.driver.session() as session:
            result = session.read_transaction(self._search_documents, keywords)
        return result

    @staticmethod
    def _search_documents(tx, keywords):
        query = """
        MATCH (d:Document)
        WHERE ANY(keyword IN $keywords WHERE toLower(d.content) CONTAINS toLower(keyword))
        RETURN d.title AS title, d.content AS content 
        """
        result = tx.run(query, keywords=keywords)
        documents = [{"title": record["title"], "content": record["content"]} for record in result]
        return documents
    
    def get_relevant_documents(self, keywords, limit):
        with self.driver.session() as session:
            result = session.read_transaction(self._get_documents_by_relevance, keywords, limit)
        return result
    
    @staticmethod
    def _get_documents_by_relevance(tx, keywords, limit):
        relevant_documents=[{}]
        keyword_phrase = ' '.join(keywords)
        query = """
        CALL db.index.fulltext.queryNodes("documentsTextIndex", $keyword_phrase) YIELD node, score
        RETURN node.title AS title, node.content AS content, node.score AS score
        ORDER BY score DESC
        LIMIT $limit
        """
        result = tx.run(query, keyword_phrase=keyword_phrase, limit=limit)
        documents = [{"title": record["title"], "content": record["content"]} for record in result]
        relevant_documents.append(documents)
        return documents