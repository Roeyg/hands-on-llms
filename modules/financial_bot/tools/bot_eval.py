import logging
import json

import fire
import pickle

from datasets import Dataset

from tools.bot import load_bot

logger = logging.getLogger(__name__)


def evaluate_w_ragas(query: str, context: list[str], output: str, ground_truth: str, metrics: list) -> dict:
    """
    Evaluate the RAG (query,context,response) using RAGAS
    """
    from ragas import evaluate
    data_sample = {
        "question": [query],  # Question as Sequence(str)
        "answer": [output],  # Answer as Sequence(str)
        "contexts": [context],  # Context as Sequence(str)
        "ground_truths": [[ground_truth]],  # Ground Truth as Sequence(str)
    }

    dataset = Dataset.from_dict(data_sample)
    score = evaluate(
        dataset=dataset,
        metrics=metrics,
    )

    return score

def run_local(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        #context_entity_recall,
        context_recall,
        #context_relevancy,
        #context_utilization,
        faithfulness
    )
    from ragas.metrics.context_precision import context_relevancy
    metrics = [
        #context_utilization,
        context_relevancy,
        context_recall,
        answer_similarity,
        #context_entity_recall,
        #answer_correctness,
        faithfulness
    ]

    with open(testset_path, "r") as f:
        data = json.load(f)
        for elem in data:
            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
                "prompt_method": "explicit_reasoning"
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)
            logger.info("Score=%s", evaluate_w_ragas(query=elem["question"], context=output_context.split('\n'), output=response, ground_truth=elem["response"], metrics=metrics))

    return response


def run_all_prompts(
    testset_path: str,
):
    """
    Run the bot locally in production or dev mode.
    With all the prompts.

    Args:
        testset_path (str): A string containing path to the testset.

    Returns:
        str: A string containing the bot's response to the user's question.
    """

    
    from financial_bot.chains import Prompt
    prompt = Prompt()
    bot = load_bot(model_cache_dir=None)
    # Import ragas only after loading the environment variables inside load_bot()
    from ragas.metrics import (
        answer_correctness,
        answer_similarity,
        #context_entity_recall,
        context_recall,
        #context_relevancy,
        #context_utilization,
        faithfulness
    )
    from ragas.metrics.context_precision import context_relevancy
    metrics = [
       #context_utilization,
        context_relevancy,
        context_recall,
        answer_similarity,
        #context_entity_recall,
        #answer_correctness,
        faithfulness
    ]
    with open(testset_path, "r") as f:
        data = json.load(f)
    for key in prompt.prompts.keys():
        if key == "no_prompt":
            continue
        all_= dict()
        import os
        dump_path = os.path.expanduser(f"~/results/{key}.pickle")
        prompt_eval = []
        for i,elem in enumerate(data):    
            print(f"############## key: {key}, i:{i}  ##################")

            input_payload = {
                "about_me": elem["about_me"],
                "question": elem["question"],
                "to_load_history": [],
                "prompt_method": key
            }
            output_context = bot.finbot_chain.chains[0].run(input_payload)
            response = bot.answer(**input_payload)
            curr_eval = evaluate_w_ragas(query=elem["question"], context=output_context.split('\n'), output=response, ground_truth=elem["response"], metrics=metrics)
            prompt_eval.append(curr_eval)
            logger.info("Score=%s", curr_eval)
        
        with open(dump_path, 'wb') as f:
            pickle.dump(prompt_eval, f)

    return response

def get_results():
    from financial_bot.chains import Prompt
    from pathlib import Path
    prompt = Prompt()
    bot = load_bot(model_cache_dir=None)
    dir_path =Path("/home/student/results/")
    for file in dir_path.iterdir():
        with open(file, "rb") as f:
            print(f"\n########################### key: {file.parts[-1]}   ###########################\n")
            data = pickle.load(f)
            print(data)

if __name__ == "__main__":
    # get_results()
    fire.Fire(run_all_prompts)
