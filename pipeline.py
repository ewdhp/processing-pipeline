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

    def get_functions(self, base_func_name):
        func_list = []
        regex = re.compile(rf"{base_func_name}(_\w+)?$")
        for module in self.additional_modules.values():
            func_list.extend([getattr(module, fn) for fn in dir(module) if regex.match(fn) and callable(getattr(module, fn))])
        return func_list

    async def run_pipeline(self, text, pipeline_config):
        tokens = text
        flow = pipeline_config.get("flow", ["preprocessing", "analysis", "postprocessing"])
        results = None
        
        for stage in flow:
            if stage in pipeline_config:
                stage_config = pipeline_config[stage]
                if stage == "preprocessing":
                    tokens = await self.run_stage(tokens, stage_config)
                elif stage == "analysis":
                    results = await self.run_stage(tokens, stage_config)
                elif stage == "postprocessing":
                    results = await self.run_stage(results, stage_config)
        
        self.config = self.load_config(self.config_file)  # Reload configuration file
        self.additional_modules = self.load_additional_modules()  # Reload additional modules
        return results

    async def run_stage(self, data, stage_config):
        tasks = []
        for step, config in stage_config.items():
            func_name = config["function"]
            functions = self.get_functions(func_name)
            if len(functions) > 1:  # Multiple functions found
                tasks.append(self.run_multiple_functions(data, functions, config.get("params", {})))
            else:
                func = functions[0]
                if config.get("async", False):
                    tasks.append(func(data, **config.get("params", {})))
                else:
                    data = func(data, **config.get("params", {}))
                    logging.info(f"Step '{func_name}' completed.")
        
        if tasks:
            results = await asyncio.gather(*tasks)
            for result in results:
                if isinstance(result, list):
                    data = self.evaluate_results(result, config.get("metrics", {}))
                else:
                    data = result
        return data

    async def run_multiple_functions(self, data, functions, params):
        tasks = [fn(data, **params) for fn in functions]
        results = await asyncio.gather(*tasks)
        return results

    def evaluate_results(self, results, metrics):
        for metric, threshold in metrics.items():
            for result in results:
                if metric == "accuracy_threshold" and result.get("accuracy", 0) < threshold:
                    return False
                if metric == "precision_threshold" and result.get("precision", 0) < threshold:
                    return False
                if metric == "sentiment_score_threshold" and result.get("sentiment_score", 0) < threshold:
                    return False
        return True

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
