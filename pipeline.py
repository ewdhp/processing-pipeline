import json
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import pandas as pd
import importlib.util
import logging
import asyncio
import glob
import re

# Configure logging
logging.basicConfig(filename='pipeline.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TextProcessingPipeline:
    def __init__(self, config_file, max_files):
        self.config_file = config_file
        self.config = self.load_config(config_file)
        self.nlp = spacy.load("en_core_web_sm")
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = TfidfVectorizer()
        self.max_retries = 3
        self.max_files = max_files
        self.additional_modules = self.load_additional_modules()

    def load_config(self, file_path):
        with open(file_path, 'r') as f:
            return json.load(f)

    def load_additional_modules(self):
        modules = {}
        for i in range(self.max_files):
            file_pattern = f'*_{i}.py'
            files = glob.glob(file_pattern)
            pattern = re.compile(r'^[a-zA-Z0-9]+_[0-9]+\.py$')
            for file in files:
                if pattern.match(file):
                    module_name = file.replace('.py', '')
                    try:
                        spec = importlib.util.spec_from_file_location(module_name, file)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)
                        modules[module_name] = module
                    except FileNotFoundError:
                        logging.warning(f"File {file} not found. Skipping.")
        return modules

    def get_function(self, func_name):
        for module in self.additional_modules.values():
            if hasattr(module, func_name):
                return getattr(module, func_name)
        raise AttributeError(f"Function '{func_name}' not found in additional modules.")

    async def run_pipeline(self, text, pipeline_config):
        tokens = text
        if "preprocessing" in pipeline_config:
            tokens = await self.preprocess(tokens, pipeline_config["preprocessing"])
        if "analysis" in pipeline_config:
            results = await self.analyze(tokens, pipeline_config["analysis"])
        if "postprocessing" in pipeline_config:
            results = await self.postprocess(results, pipeline_config["postprocessing"])
        
        self.config = self.load_config(self.config_file)  # Reload configuration file
        self.additional_modules = self.load_additional_modules()  # Reload additional modules
        return results

    async def preprocess(self, tokens, preprocess_config):
        tasks = []
        for step, config in preprocess_config.items():
            func_name = config["function"]
            func = self.get_function(func_name)
            if config.get("async", False):
                tasks.append(func(tokens, **config.get("params", {})))
            else:
                tokens = func(tokens, **config.get("params", {}))
                logging.info(f"Preprocessing step '{func_name}' completed.")
        
        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                tokens = result

        return tokens

    async def analyze(self, tokens, analysis_config):
        tasks = []
        results = {}
        for task_config in analysis_config:
            task = task_config["task"]
            params = task_config.get("params", {})
            metrics = task_config.get("metrics", {})
            func = self.get_function(task)
            if task_config.get("async", False):
                tasks.append(func(tokens, **params))
            else:
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
        
        if tasks:
            async_results = await asyncio.gather(*tasks)
            for async_result in async_results:
                results.update(async_result)

        return results

    async def postprocess(self, results, postprocess_config):
        tasks = []
        for step, config in postprocess_config.items():
            func_name = config["function"]
            params = config.get("params", {})
            func = self.get_function(func_name)
            if config.get("async", False):
                tasks.append(func(results, **params))
            else:
                results = func(results, **params)
                logging.info(f"Postprocessing step '{func_name}' completed.")
        
        if tasks:
            results = await asyncio.gather(*tasks)

        return results

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

    async def run(self, text, config=None):
        if config is None:
            config = self.config
        for pipeline_name, pipeline_config in config["pipelines"].items():
            logging.info(f"Running {pipeline_name}...")
            results = await self.run_pipeline(text, pipeline_config)
            logging.info(f"Completed {pipeline_name}.")
            if "nested_pipelines" in pipeline_config:
                await self.run(text, pipeline_config["nested_pipelines"])

# Load configuration
config_file = 'config.json'
config = json.load(open(config_file))

# Sample text
text = "The quick brown fox jumps over the lazy dog."

# Initialize and run pipeline
pipeline = TextProcessingPipeline(config_file, max_files=10)  # Set your max value here
asyncio.run(pipeline.run(text))
