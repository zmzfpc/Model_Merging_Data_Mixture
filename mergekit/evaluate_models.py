#!/usr/bin/env python3
"""
Comprehensive evaluation script using lm-evaluation-harness and custom metrics
for testing merged models on code-to-text and other tasks.
"""

import os
import sys
import json
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, results_dir="evaluation_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        # Standard evaluation tasks
        self.tasks = {
            'code_generation': ['humaneval', 'mbpp'],
            'general_nlp': ['hellaswag', 'arc_easy', 'arc_challenge', 'winogrande'],
            'reasoning': ['gsm8k', 'mathqa'],
            'code_understanding': ['conala', 'spider']  # if available
        }
        
        # Custom evaluation settings
        self.eval_settings = {
            'batch_size': 'auto',
            'max_batch_size': 64,
            'device': 'cuda',
            'dtype': 'bfloat16',
        }
    
    def check_lm_eval_installation(self):
        """Check if lm-evaluation-harness is installed."""
        try:
            result = subprocess.run(['lm_eval', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                logger.info("lm-evaluation-harness is available")
                return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass
        
        logger.warning("lm-evaluation-harness not found. Installing...")
        return self.install_lm_eval()
    
    def install_lm_eval(self):
        """Install lm-evaluation-harness."""
        try:
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', 
                'lm-eval[vllm]', '--upgrade'
            ], check=True)
            logger.info("Successfully installed lm-evaluation-harness")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install lm-evaluation-harness: {e}")
            return False
    
    def run_lm_eval(self, model_path, tasks, output_file):
        """Run lm-evaluation-harness on specified tasks."""
        cmd = [
            'lm_eval',
            '--model', 'vllm',
            '--model_args', f'pretrained={model_path},dtype={self.eval_settings["dtype"]},gpu_memory_utilization=0.8',
            '--tasks', ','.join(tasks),
            '--batch_size', str(self.eval_settings['batch_size']),
            '--output_path', str(output_file),
            '--log_samples'
        ]
        
        logger.info(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
            
            if result.returncode == 0:
                logger.info(f"Successfully completed evaluation for tasks: {tasks}")
                return True, result.stdout
            else:
                logger.error(f"Evaluation failed: {result.stderr}")
                return False, result.stderr
                
        except subprocess.TimeoutExpired:
            logger.error("Evaluation timed out after 1 hour")
            return False, "Timeout"
        except Exception as e:
            logger.error(f"Error running evaluation: {e}")
            return False, str(e)
    
    def run_evalplus(self, model_path, dataset='humaneval'):
        """Run EvalPlus evaluation."""
        cmd = [
            'evalplus.evaluate',
            '--model', model_path,
            '--backend', 'vllm',
            '--dataset', dataset,
            '--greedy'
        ]
        
        logger.info(f"Running EvalPlus: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            return result.returncode == 0, result.stdout
        except Exception as e:
            logger.error(f"EvalPlus failed: {e}")
            return False, str(e)
    
    def run_code_summarization(self, model_path):
        """Run custom code summarization evaluation."""
        test_data = "/dccstor/unified-trans/model_merging/granite33_2/LLaMA-Factory/data/instruct_code_docstring_train/test.jsonl"
        side_model = "sentence-transformers/all-mpnet-base-v2"
        
        if not Path(test_data).exists():
            logger.warning(f"Test data not found: {test_data}")
            return False, "Test data not found"
        
        output_file = self.results_dir / f"{Path(model_path).name}_codesum.jsonl"
        
        cmd = [
            'python', 'eval_codesum.py',
            '--model', model_path,
            '--data', test_data,
            '--side', side_model,
            '--batch_size', '64',
            '--bf16', '--vllm',
            '--save_path', str(output_file)
        ]
        
        logger.info(f"Running code summarization: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
            return result.returncode == 0, result.stdout
        except Exception as e:
            logger.error(f"Code summarization failed: {e}")
            return False, str(e)
    
    def parse_lm_eval_results(self, output_file):
        """Parse lm-evaluation-harness results."""
        results = {}
        
        if not output_file.exists():
            return results
        
        try:
            with open(output_file, 'r') as f:
                data = json.load(f)
            
            # Extract results for each task
            if 'results' in data:
                for task, metrics in data['results'].items():
                    results[task] = {}
                    for metric, value in metrics.items():
                        if isinstance(value, (int, float)):
                            results[task][metric] = value
                        elif isinstance(value, dict) and 'exact_match' in value:
                            results[task][metric] = value['exact_match']
            
        except Exception as e:
            logger.error(f"Error parsing results from {output_file}: {e}")
        
        return results
    
    def parse_evalplus_output(self, output):
        """Parse EvalPlus output."""
        results = {}
        
        lines = output.split('\n')
        for line in lines:
            if 'pass@1:' in line:
                parts = line.split()
                if 'humaneval' in line.lower():
                    if '+' in line:
                        results['humaneval_plus'] = float(parts[-1])
                    else:
                        results['humaneval_base'] = float(parts[-1])
                elif 'mbpp' in line.lower():
                    if '+' in line:
                        results['mbpp_plus'] = float(parts[-1])
                    else:
                        results['mbpp_base'] = float(parts[-1])
        
        return results
    
    def evaluate_model(self, model_path, model_name=None):
        """Comprehensive evaluation of a single model."""
        if model_name is None:
            model_name = Path(model_path).name
        
        logger.info(f"Starting evaluation of model: {model_name}")
        
        # Check if model exists
        if not Path(model_path).exists():
            logger.error(f"Model path does not exist: {model_path}")
            return None
        
        model_results = {
            'model_name': model_name,
            'model_path': model_path,
            'evaluation_date': datetime.now().isoformat(),
            'tasks': {}
        }
        
        # 1. Run lm-evaluation-harness tasks
        if self.check_lm_eval_installation():
            for category, tasks in self.tasks.items():
                output_file = self.results_dir / f"{model_name}_{category}_results.json"
                
                success, output = self.run_lm_eval(model_path, tasks, output_file)
                
                if success:
                    parsed_results = self.parse_lm_eval_results(output_file)
                    model_results['tasks'][category] = parsed_results
                else:
                    logger.warning(f"Failed to evaluate {category}: {output}")
                    model_results['tasks'][category] = {'error': output}
        
        # 2. Run EvalPlus evaluations  
        for dataset in ['humaneval', 'mbpp']:
            success, output = self.run_evalplus(model_path, dataset)
            if success:
                evalplus_results = self.parse_evalplus_output(output)
                model_results['tasks'][f'evalplus_{dataset}'] = evalplus_results
            else:
                logger.warning(f"EvalPlus {dataset} failed: {output}")
        
        # 3. Run code summarization
        success, output = self.run_code_summarization(model_path)
        if success:
            model_results['tasks']['code_summarization'] = {'status': 'completed'}
        else:
            logger.warning(f"Code summarization failed: {output}")
            model_results['tasks']['code_summarization'] = {'error': output}
        
        # Save individual model results
        results_file = self.results_dir / f"{model_name}_results.json"
        with open(results_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        
        logger.info(f"Evaluation completed for {model_name}")
        return model_results
    
    def evaluate_all_models(self, models_dir="merged_model"):
        """Evaluate all models in the specified directory."""
        models_dir = Path(models_dir)
        
        if not models_dir.exists():
            logger.error(f"Models directory does not exist: {models_dir}")
            return []
        
        # Find all model directories
        model_paths = [p for p in models_dir.iterdir() if p.is_dir()]
        
        if not model_paths:
            logger.warning(f"No models found in {models_dir}")
            return []
        
        logger.info(f"Found {len(model_paths)} models to evaluate")
        
        all_results = []
        
        for model_path in model_paths:
            try:
                results = self.evaluate_model(str(model_path), model_path.name)
                if results:
                    all_results.append(results)
            except Exception as e:
                logger.error(f"Error evaluating {model_path}: {e}")
                continue
        
        # Create consolidated results
        self.create_summary_report(all_results)
        
        return all_results
    
    def create_summary_report(self, all_results):
        """Create a summary report of all evaluations."""
        if not all_results:
            logger.warning("No results to summarize")
            return
        
        # Create CSV summary
        summary_data = []
        
        for result in all_results:
            row = {
                'model_name': result['model_name'],
                'evaluation_date': result['evaluation_date']
            }
            
            # Extract key metrics
            tasks = result.get('tasks', {})
            
            # EvalPlus results
            for dataset in ['humaneval', 'mbpp']:
                evalplus_key = f'evalplus_{dataset}'
                if evalplus_key in tasks:
                    evalplus_data = tasks[evalplus_key]
                    row[f'{dataset}_base'] = evalplus_data.get(f'{dataset}_base', '')
                    row[f'{dataset}_plus'] = evalplus_data.get(f'{dataset}_plus', '')
            
            # LM-eval results (extract key metrics)
            for category, task_results in tasks.items():
                if category.startswith('evalplus_') or category == 'code_summarization':
                    continue
                
                for task, metrics in task_results.items():
                    if isinstance(metrics, dict):
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                row[f'{task}_{metric}'] = value
            
            summary_data.append(row)
        
        # Save CSV summary
        if summary_data:
            df = pd.DataFrame(summary_data)
            summary_file = self.results_dir / 'evaluation_summary.csv'
            df.to_csv(summary_file, index=False)
            logger.info(f"Summary report saved to: {summary_file}")
        
        # Save detailed JSON report
        detailed_file = self.results_dir / 'detailed_results.json'
        with open(detailed_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        logger.info(f"Detailed results saved to: {detailed_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate merged models using multiple metrics')
    parser.add_argument('--model', type=str, help='Path to specific model to evaluate')
    parser.add_argument('--models_dir', type=str, default='merged_model', 
                       help='Directory containing models to evaluate')
    parser.add_argument('--results_dir', type=str, default='evaluation_results',
                       help='Directory to save results')
    parser.add_argument('--tasks', type=str, nargs='+', 
                       choices=['code_generation', 'general_nlp', 'reasoning', 'code_understanding'],
                       help='Specific task categories to run')
    
    args = parser.parse_args()
    
    evaluator = ModelEvaluator(results_dir=args.results_dir)
    
    # Filter tasks if specified
    if args.tasks:
        evaluator.tasks = {k: v for k, v in evaluator.tasks.items() if k in args.tasks}
    
    if args.model:
        # Evaluate single model
        results = evaluator.evaluate_model(args.model)
        if results:
            print(f"Evaluation completed for {args.model}")
            print(f"Results saved in {args.results_dir}")
    else:
        # Evaluate all models
        results = evaluator.evaluate_all_models(args.models_dir)
        print(f"Evaluated {len(results)} models")
        print(f"Results saved in {args.results_dir}")

if __name__ == "__main__":
    main()
