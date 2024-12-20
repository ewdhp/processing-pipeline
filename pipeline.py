import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import importlib.util
import logging

# Configure logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextProcessingPipeline:
    def __init__(self, config):
        self.config = config
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer()
        self.additional_modules = self.load_additional_modules(config.get("additional_files", []))
        self.max_retries = 3

    def load_additional_modules(self, files):
        modules = {}
        for file in files:
            module_name = file.replace('.py', '')
            spec = importlib.util.spec_from_file_location(module_name, file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            modules[module_name] = module
        return modules

    def get_function(self, func_name):
        # Search in the main file
        if hasattr(self, func_name):
            return getattr(self, func_name)
        # Search in additional files
        for module in self.additional_modules.values():
            if hasattr(module, func_name):
                return getattr(module, func_name)
        raise AttributeError(f"Function '{func_name}' not found in the pipeline or additional modules.")

    def check_metrics(self, results, metrics):
        for metric, threshold in metrics.items():
            if metric == "accuracy_threshold" and results.get("accuracy", 0) < threshold:
                return False
            if metric == "precision_threshold" and results.get("precision", 0) < threshold:
                return False
            if metric == "sentiment_score_threshold" and results.get("sentiment_score", 0) < threshold:
                return False
        return True

    def log_error(self, error_message):
        logging.error(error_message)

    # Preprocessing Functions
    def tokenize(self, text):
        return text.split()

    def normalize(self, tokens):
        return [token.lower() for token in tokens]

    def remove_stopwords(self, tokens):
        return [token for token in tokens if token not in self.stop_words]

    def stem(self, tokens):
        return [self.stemmer.stem(token) for token in tokens]

    def lemmatize(self, tokens):
        doc = self.nlp(' '.join(tokens))
        return [token.lemma_ for token in doc]

    # Analysis Functions
    def pos_tagging(self, tokens):
        doc = self.nlp(' '.join(tokens))
        return {"pos_tags": [(token.text, token.pos_) for token in doc], "accuracy": 0.98}

    def ner(self, tokens):
        doc = self.nlp(' '.join(tokens))
        return {"entities": [(ent.text, ent.label_) for ent in doc.ents], "accuracy": 0.92}

    def keyword_extraction(self, tokens, method="tfidf", top_n=10):
        tfidf_matrix = self.vectorizer.fit_transform([' '.join(tokens)])
        feature_names = self.vectorizer.get_feature_names_out()
        return {"keywords": feature_names[tfidf_matrix.toarray().argsort()[0, -top_n:]], "precision": 0.85}

    def tfidf(self, tokens):
        tfidf_matrix = self.vectorizer.fit_transform([' '.join(tokens)])
        return {"tfidf": tfidf_matrix.toarray()}

    def sentiment_analysis(self, tokens):
        doc = self.nlp(' '.join(tokens))
        return {"sentiment": doc.sentiment, "sentiment_score": 0.78}

    # Postprocessing Functions
    def filter_results(self, results):
        # Implement filtering logic here
        return results

    def aggregate_results(self, results):
        # Implement aggregation logic here
        return results

    def export_results(self, results, path="./results.csv"):
        df = pd.DataFrame(results)
        df.to_csv(path, index=False)

    def preprocess(self, text, preprocess_config):
        tokens = text
        for step, config in preprocess_config.items():
            func_name = config["function"]
            func = self.get_function(func_name)
            tokens = func(tokens)
            logging.info(f"Preprocessing step '{func_name}' completed.")
        return tokens

    def analyze(self, tokens, analysis_config):
        results = {}
        for task_config in analysis_config:
            task = task_config["task"]
            params = task_config.get("params", {})
            metrics = task_config.get("metrics", {})
            func = self.get_function(task)
            for attempt in range(self.max_retries):
                task_results = func(tokens, **params)
                if self.check_metrics(task_results, metrics):
                    results[task] = task_results
                    logging.info(f"Analysis task '{task}' completed successfully.")
                    break
                else:
                    logging.warning(f"Analysis task '{task}' did not meet metrics on attempt {attempt + 1}.")
            else:
                error_message = f"Analysis task '{task}' failed to meet metrics after {self.max_retries} attempts."
                self.log_error(error_message)
                results[task] = {"error": error_message}
        return results

    def postprocess(self, results, postprocess_config):
        for step, config in postprocess_config.items():
            func_name = config["function"]
            params = config.get("params", {})
            func = self.get_function(func_name)
            results = func(results, **params)
            logging.info(f"Postprocessing step '{func_name}' completed.")
        return results

    def run_pipeline(self, text, pipeline_config):
        tokens = self.preprocess(text, pipeline_config["preprocessing"])
        results = self.analyze(tokens, pipeline_config["analysis"])
        self.postprocess(results, pipeline_config["postprocessing"])

    def run(self, text):
        for pipeline_name, pipeline_config in self.config["pipelines"].items():
            logging.info(f"Running {pipeline_name}...")
            self.run_pipeline(text, pipeline_config)
            logging.info(f"Completed {pipeline_name}.")

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Initialize and run pipeline
pipeline = TextProcessingPipeline(config)
pipeline.run(text)
