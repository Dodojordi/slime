# import asyncio

# import aiohttp

# from slime.utils.misc import load_function
# from slime.utils.types import Sample

# from .deepscaler import get_deepscaler_rule_based_reward
# from .f1 import f1_score
# from .gpqa import compute_gpqa_reward
# from .math_dapo_utils import compute_score as compute_score_dapo
# from .math_utils import extract_answer as extract_boxed_answer
# from .math_utils import grade_answer_verl
# from .p1 import compute_score_p1

# async def remote_rm(args, sample: Sample, is_evaluation):
#     if "</think>" in sample.response:
#         response = sample.response.split("</think>")[-1]
#     else:
#         response = sample.response
#     # print(f"sample.label: {sample.label}, len(sample.label): {len(sample.label)}")
    
#     if "is_proof" in sample.metadata:
#         payload = {
#             "prompt": sample.prompt,
#             "response": response,
#             "label": (
#                 [sample.label] if isinstance(sample.label, str) else sample.label
#                 if isinstance(sample.label, list) else None
#             ),
#             "points": (
#                 sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
#                 and all(isinstance(p, float) for p in sample.metadata.get("points", []))
#                 else None
#             ),
#             "question": (
#                 sample.metadata.get("question", None) if isinstance(sample.metadata.get("question", None), str) else None
#             ),
#             "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify,
#             "is_proof": sample.metadata.get("is_proof", False)
#         }
#     else:
#         payload = {
#             "prompt": sample.prompt,
#             "response": response,
#             "label": (
#                 [sample.label] if isinstance(sample.label, str) else sample.label
#                 if isinstance(sample.label, list) else None
#             ),
#             "points": (
#                 sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
#                 and all(isinstance(p, float) for p in sample.metadata.get("points", []))
#                 else None
#             ),
#             "question": (
#                 sample.metadata.get("question", None) if isinstance(sample.metadata.get("question", None), str) else None
#             ),
#             "use_xverify": args.eval_use_xverify if is_evaluation else args.train_use_xverify
#         }
    
    
#     # Retry configuration
#     max_retries = 3
#     base_delay = 1.0  # seconds
    
#     for attempt in range(max_retries):
#         try:
#             session_kwargs = {}
#             async with aiohttp.ClientSession(**session_kwargs) as session:
#                 async with session.post(args.rm_url, json=payload) as resp:
#                     if resp.status == 200:
#                         return await resp.json()
#                     else:
#                         # Get the error response for debugging
#                         try:
#                             error_text = await resp.text()
#                             print(f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: {error_text}")
#                             print(f"Payload sent: {payload}")
#                         except:
#                             print(f"Remote RM server error (status {resp.status}) on attempt {attempt + 1}/{max_retries}: Unable to read error response")
#                             print(f"Payload sent: {payload}")
                        
#                         if attempt < max_retries - 1:
#                             delay = base_delay * (2 ** attempt)  # Exponential backoff
#                             print(f"Retrying in {delay} seconds...")
#                             await asyncio.sleep(delay)
#                         else:
#                             # Final attempt failed, return default
#                             print(f"All {max_retries} attempts failed. Returning default reward value.")
#                             return {
#                                 "score": 0.0,
#                                 "point": 0.0,
#                                 "acc": False,
#                                 "extracted_gt": "",
#                                 "extracted_pred": "",
#                                 "scored_by": "default_fallback",
#                                 "score_noxverify": 0.0,
#                                 "point_noxverify": 0.0,
#                             }
#         except Exception as e:
#             print(f"Network error on attempt {attempt + 1}/{max_retries}: {str(e)}")
#             if attempt < max_retries - 1:
#                 delay = base_delay * (2 ** attempt)  # Exponential backoff
#                 print(f"Retrying in {delay} seconds...")
#                 await asyncio.sleep(delay)
#             else:
#                 # Final attempt failed, return default
#                 print(f"All {max_retries} attempts failed due to network errors. Returning default reward value.")
#                 return {
#                     "score": 0.0,
#                     "point": 0.0,
#                     "acc": False,
#                     "extracted_gt": "",
#                     "extracted_pred": "",
#                     "scored_by": "default_fallback",
#                     "score_noxverify": 0.0,
#                     "point_noxverify": 0.0,
#                 }


# async def async_rm(args, sample: Sample, **kwargs):
#     if args.custom_rm_path is not None:
#         rm_function = load_function(args.custom_rm_path)
#         return await rm_function(args, sample, **kwargs)

#     is_evalution = kwargs.get("evaluation", False)
    
#     metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
#     rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
#     response = sample.response
#     label = sample.label
#     if rm_type.startswith("boxed_"):
#         response = extract_boxed_answer(response) or ""
#         rm_type = rm_type[len("boxed_") :]

#     # This function is intended for remote or time-consuming reward model evaluation.
#     # Implement the actual logic as needed.
#     if rm_type == "remote_rm":
#         return await remote_rm(args, sample, is_evalution)
#     elif rm_type == "p1":
#         return compute_score_p1(response, 
#             label if isinstance(label, list) else [label], 
#             points=sample.metadata.get("points", None) if isinstance(sample.metadata.get("points", None), list)
#             and all(isinstance(p, float) for p in sample.metadata.get("points", []))
#             else None,
#             use_xverify=args.eval_use_xverify if is_evalution else args.train_use_xverify, 
#             base_url=args.rm_url,
#             question=sample.metadata.get("question", None) if isinstance(sample.metadata.get("question", None), str) else None
#         )
#     elif rm_type == "deepscaler":
#         return get_deepscaler_rule_based_reward(response, label)
#     elif rm_type == "dapo":
#         return compute_score_dapo(response, label)
#     elif rm_type == "math":
#         return 1 if grade_answer_verl(response, label) else 0
#     elif rm_type == "f1":
#         return f1_score(response, label)[0]
#     elif rm_type == "gpqa":
#         return compute_gpqa_reward(response, label, metadata=metadata)
#     elif rm_type == "ifbench":
#         from .ifbench import compute_ifbench_reward

#         return compute_ifbench_reward(response, label, metadata=metadata)
#     elif rm_type:
#         raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
#     else:
#         raise NotImplementedError("Rule-based RM type is not specified.")


# async def batched_async_rm(
#     args,
#     samples: list[Sample],
#     **kwargs,
# ) -> list[int | float]:
#     if args.custom_rm_path is not None:
#         # Ensure the custom reward function is implemented in batch mode
#         rm_function = load_function(args.custom_rm_path)
#         return await rm_function(args, samples, **kwargs)
#     tasks = [async_rm(args, sample, **kwargs) for sample in samples]
#     rewards = await asyncio.gather(*tasks)
#     return rewards
import asyncio
import random

import aiohttp

from slime.utils.misc import load_function
from slime.utils.types import Sample

from .deepscaler import get_deepscaler_rule_based_reward
from .f1 import f1_score
from .gpqa import compute_gpqa_reward
from .math_dapo_utils import compute_score as compute_score_dapo
from .math_utils import extract_answer as extract_boxed_answer
from .math_utils import grade_answer_verl


async def remote_rm(args, sample: Sample):
    payload = {
        "prompt": sample.prompt,
        "response": sample.response,
        "label": sample.label,
    }
    session_kwargs = {}
    async with aiohttp.ClientSession(**session_kwargs) as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def async_rm(args, sample: Sample, **kwargs):
    if args.custom_rm_path is not None:
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, sample, **kwargs)

    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    rm_type = (metadata.get("rm_type") or args.rm_type or "").strip()
    response = sample.response
    label = sample.label
    if rm_type.startswith("boxed_"):
        response = extract_boxed_answer(response) or ""
        rm_type = rm_type[len("boxed_") :]

    # This function is intended for remote or time-consuming reward model evaluation.
    # Implement the actual logic as needed.
    if rm_type == "remote_rm":
        return await remote_rm(args, sample)
    elif rm_type == "deepscaler":
        return get_deepscaler_rule_based_reward(response, label)
    elif rm_type == "dapo":
        return compute_score_dapo(response, label,True)
    elif rm_type == "math":
        return 1 if grade_answer_verl(response, label) else 0
    elif rm_type == "f1":
        return f1_score(response, label)[0]
    elif rm_type == "gpqa":
        return compute_gpqa_reward(response, label, metadata=metadata)
    elif rm_type == "ifbench":
        from .ifbench import compute_ifbench_reward

        return compute_ifbench_reward(response, label, metadata=metadata)
    elif rm_type == "random":
        return random.randint(0, 1)
    elif rm_type:
        raise NotImplementedError(f"Rule-based RM for {rm_type} is not implemented.")
    else:
        raise NotImplementedError("Rule-based RM type is not specified.")


async def batched_async_rm(
    args,
    samples: list[Sample],
    **kwargs,
) -> list[int | float]:
    if args.custom_rm_path is not None:
        # Ensure the custom reward function is implemented in batch mode
        rm_function = load_function(args.custom_rm_path)
        return await rm_function(args, samples, **kwargs)
    tasks = [async_rm(args, sample, **kwargs) for sample in samples]
    rewards = await asyncio.gather(*tasks)
    return rewards
