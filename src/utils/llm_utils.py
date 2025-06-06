import re
import os
from dotenv import load_dotenv, find_dotenv

from llama_index.llms.openai import OpenAI
from vllm.entrypoints.llm import LLM
from vllm.sampling_params import SamplingParams
from huggingface_hub.hf_api import HfFolder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import pynvml
IS_DEBUG_GPU = False
def print_gpu_utilization(info=""):
    def _convert_bytes(num_bytes):
        for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
            if num_bytes < 1024:
                return f"{num_bytes:.2f} {unit}"
            num_bytes /= 1024
    if IS_DEBUG_GPU:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)  # GPU 0
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        total_memory = _convert_bytes(meminfo.total)
        free_memory = _convert_bytes(meminfo.free)
        used_memory = _convert_bytes(meminfo.used)

        print("---------------------------------------")
        print(info)
        print(f"    Total memory: {total_memory}")
        print(f"    Free memory: {free_memory}")
        print(f"    Used memory: {used_memory}")
        print("---------------------------------------")
        pynvml.nvmlShutdown()


class LLMAgent:
    def __init__(self, llm="4o-mini",
                 quantization="",
                 gpu_memory_utilization=0.8,
                 temperature=0,
                 top_p=1.0,
                 max_tokens=8000,
                 gpu_count=2,
                 swap_space=4):
        self.model_name = llm
        self.msg = None
        self.key = None
        self.msg: str
        if llm == "3.5":
            self.agent = OpenAI(api_key=self.get_openai_api_key(), model="gpt-3.5-turbo", temperature=0)
        elif llm == "4":
            self.agent = OpenAI(api_key=self.get_openai_api_key(), model="gpt-4", temperature=0)
        elif llm == "4o":
            self.agent = OpenAI(api_key=self.get_openai_api_key(), model="gpt-4o", temperature=0)
        elif llm == "4o-mini":
            self.agent = OpenAI(api_key=self.get_openai_api_key(), model="gpt-4o-mini", temperature=0)
        else:
            self.load_huggingface_token()
            if quantization:
                self.agent = LLM(llm,
                          tensor_parallel_size=gpu_count,
                          gpu_memory_utilization=gpu_memory_utilization,
                          trust_remote_code=True,
                          quantization=quantization)
            elif "meta-llama/Llama-3" in llm:
                print("Special setup for llama-3 8 B")
                self.agent = LLM(model=llm,
                                 max_model_len=38000,
                                 dtype="auto")
                self.sampling_params = SamplingParams(temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens if "deepseek" in llm else 500,
                                    min_tokens=20)
            elif "TsinghuaC3I" in llm:
                print("Special setup for UltraMedical")
                self.agent = LLM(
                model="TsinghuaC3I/Llama-3-70B-UltraMedical",
                trust_remote_code=True,
                dtype=torch.bfloat16,            # compute in BF16 (or torch.float16)
                load_format="bitsandbytes",
                quantization="bitsandbytes"      # load weights as 4-bit
                )
            elif "Baichuan" in llm:
                self.tokenizer = AutoTokenizer.from_pretrained(llm,trust_remote_code=True)
                # self.agent = 
                self.agent = AutoModelForCausalLM.from_pretrained(llm,trust_remote_code=True,torch_dtype = torch.bfloat16).cuda()
            else:
                self.agent = LLM(llm,
                          tensor_parallel_size=gpu_count,
                          dtype="bfloat16",
                          swap_space=swap_space,
                          gpu_memory_utilization=gpu_memory_utilization,
                          trust_remote_code=True,
                          enforce_eager=True,
                          max_model_len=14000)
                self.sampling_params = SamplingParams(temperature=temperature,
                                                top_p=top_p,
                                                max_tokens=max_tokens if "deepseek" in llm else 500,
                                                min_tokens=20)

    def get_openai_api_key(self):
        _ = load_dotenv(find_dotenv())
        self.key = os.getenv("OPENAI_API_KEY")
        return self.key

    def load_huggingface_token(self):
        _ = load_dotenv(find_dotenv())
        token = os.getenv("HF_TOKEN")
        HfFolder.save_token(token)

    def send_msg(self, content):
        print_gpu_utilization("Before send msg")
        self.msg = content
        llm_response = FakeResponse()
        if self.model_name in ['3.5','4o','4', '4o-mini']:
            try:
                llm_response = self.agent.complete(content)
                return llm_response
            except self.agent.Error as e:
                # Handle known OpenAI errors
                print(f"An OpenAI API error occurred: {e}")
            except Exception as e:
                # Handle other potential errors
                print(f"An unexpected error occurred: {e}")
            finally:
                # This code executes regardless of earlier exceptions
                return llm_response
        elif "TsinghuaC3I" in self.model_name:
            tokenizer = AutoTokenizer.from_pretrained(
                    "TsinghuaC3I/Llama-3-70B-UltraMedical",
                    use_fast=True
                )
            sampling_params = SamplingParams(
                    temperature=0.7,
                    top_p=0.9,
                    max_tokens=1024,
                    stop=["<|eot_id|>"]
                )
            messages = [
                {"role": "user", "content": "Your final answer should have letter(s) of option only, without content of any options. "+content},
            ]
            prompts = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            outputs = self.agent.generate(prompts=prompts, sampling_params=sampling_params)
            return outputs[0].outputs[0]
        elif "Baichuan" in self.model_name:
            messages = [
                {"role": "user", "content": "Your final answer should have letter(s) of option only, without content of any options. "+content},
            ]
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.agent.device)

            # 4. Generate text
            generated_ids = self.agent.generate(
                **model_inputs,
                max_new_tokens=1024
            )
            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            # 5. Decode the generated text
            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
            response_class = FakeResponse(input=response)
            return response_class

        else:
            try:
                # llm_response = self.agent.generate(prompts=content,
                #                                    sampling_params=self.sampling_params)[0].outputs[0]
                llm_response = self.agent.generate(prompts=content,
                                                   sampling_params=self.sampling_params)
                llm_response = llm_response[0].outputs[0]
            except Exception as e:
                # Handle other potential errors
                print(f"An unexpected error occurred: {e}")
            finally:
                # This code executes regardless of earlier exceptions
                print_gpu_utilization("Got reply")
                return llm_response

    def extract_answer_and_reason(self, input_string, dataset="EHRNoteQA"):
        """
        Extracts the answer and reason from a given string using regular expressions.

        Args:
            input_string (str): The string containing the 'ANSWER' and 'REASON'.
            dataset (str): The dataset name to fit different

        Returns:
            tuple: A tuple containing the answer letter and the reason text.
        """
        # Regular expression to find the answer and the reason
        if dataset == "EHRNoteQA":
            answer_pattern = r'"ANSWER"\s*:\s*([A-Z])'
            reason_pattern = r'"REASON"\s*:\s*(.*)'
        elif dataset == "benchmark_medications" or dataset == "benchmark_diagnosis" or dataset == "benchmark_choice":
            answer_pattern = r'"*ANSWER"*\s*:\s*(.*)'
            reason_pattern = r'"*REASON"*\s*:\s*(.*?)\s*(?="*ANSWER"*)'
        else:
            answer_pattern = r'"ANSWER"\s*:\s*(.*)'
            reason_pattern = r'"REASON"\s*:\s*(.*?)\s*(?="ANSWER")'

        # Searching for patterns in the input string
        answer_match = re.search(answer_pattern, input_string)
        reason_match = re.search(reason_pattern, input_string)

        # Extracting the answer and the reason if matches are found
        answer = answer_match.group(1) if answer_match else None
        reason = reason_match.group(1) if reason_match else None

        # Try one more time
        if answer is None:
            another_answer_pattern = r'\**ANSWER\**:\s*([A-Z]+)'
            answer_match = re.search(another_answer_pattern, input_string)
            answer = answer_match.group(1) if answer_match else None
        
        # Try one more time
        if answer is None:
            another_answer_pattern = r':\s*([A-Za-z]+)(?:\n|$)'
            answer_match = re.search(another_answer_pattern, input_string)
            answer = answer_match.group(1) if answer_match else None

        if answer and "<EOS_TOKEN>" in answer: 
            answer = answer.replace("<EOS_TOKEN>", "")
        return answer, reason

    def filter_sql_blocks(self, input_text):
        try:
            sql = str(self.extract_code_blocks(input_text, "sql"))
        except:
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print("WARNING: SQL can not be extracted, process again")
            sql = input_text
            # Find the index of the first occurrence of "```sql"
            start_index = sql.find("```sql")

            # Find the index of the closing "```"
            end_index = sql.find("```", start_index + len("```sql"))

            # Extract the substring between "```sql" and "```"
            sql = str(sql[start_index + len("```sql"):end_index])

        if sql.startswith("[") and sql.endswith("]"):
            # Remove the first 2 and the last 2 characters
            sql = str(sql).replace("\\n", " ")
            sql = str(sql).replace("\n", " ")
            return sql[2:-2]
        return str(sql)

    def extract_code_blocks(self, input_text, language=None):
        """
        Extracts code blocks from the provided text. Supports filtering by language.

        Args:
            input_text (str): The text containing code blocks marked with triple backticks.
            language (str, optional): The specific language to filter by (e.g., 'sql', 'python').
                                      If None, extracts all code blocks regardless of language.

        Returns:
            list of tuples: A list where each tuple contains the language and the extracted code block.
        """
        if language:
            # Regex to match code blocks of a specific language
            pattern = rf"```{language}\s(.*?)```"
        else:
            # Regex to match all code blocks with any language label
            pattern = r"```(\w+)\s(.*?)```"

        matches = re.findall(pattern, input_text, re.DOTALL)
        if not language:
            # Return a list of tuples (language, code block) when no specific language is specified
            return [(match[0], match[1].strip()) for match in matches]
        else:
            # Return a list of code blocks for the specific language
            return [match.strip() for match in matches]


class FakeResponse:
    def __init__(self, input=""):
        self.text = input
class DummyAgent:
    def __init__(self):
        self.system_prompt = ""

if __name__ == "__main__":
    # llm_agent = LLMAgent(llm="deepseek-ai/DeepSeek-R1-Distill-Llama-8B", gpu_memory_utilization=0.8, swap_space=12)
    # llm_agent = LLMAgent(llm="meta-llama/Llama-3.1-8B", gpu_memory_utilization=0.8, swap_space=12)
    # llm_agent = LLMAgent(llm="TsinghuaC3I/Llama-3-70B-UltraMedical")
    # llm_agent = LLMAgent(llm="baichuan-inc/Baichuan-M1-14B-Instruct")
    llm_agent = LLMAgent(llm="4o-mini")
    test_function = "msg" # "extract" "sql"

    test_content = '''
Answer the question below about a target patient and give me the solution of your choice.
All related info retrieved from EHR data from other most similar patients as that patient:
-----info retrieved starts here-----
The best discharge instructions for a patient who has undergone Percutaneous Coronary Intervention and has a history of unstable angina include ensuring the patient continues taking antiplatelet medications like Clopidogrel for at least one month to prevent in-stent thrombosis. Additionally, starting a statin for secondary prevention in the context of known Coronary Artery Disease is important. Close follow-up with the cardiologist is recommended, along with specific medication instructions tailored to the patient\'s condition.
-----info retrieved ends here-----

Background of that target patient:A 79-year-old male with a history of unstable angina underwent Percutaneous Coronary Intervention with the placement of a bare metal stent in the first obtuse marginal branch of the left coronary artery. He was discharged with a plan for close follow-up and specific medication instructions.

Question:What are the best discharge instructions for a patient who has undergone a Percutaneous Coronary Intervention and has a history of unstable angina?

Choices:
A: Please continue to take your medications as usual. Limit sugar intake to 50g/day. Increase fiber intake to 30g/day. Call your doctor for weight change greater than three pounds.\nB: Please continue to take your medications as usual. Please eat a diet containing less than 3g/day of sodium. Please drink less than 3L of water a day. Call your doctor for weight change greater than three pounds.\nC: Please continue to take your medications as usual. Please eat a diet containing less than 2g/day of sodium. Please drink less than 2L of water a day. Call your doctor for weight change greater than three pounds.\nD: Please continue to take your medications as usual. Please eat a diet containing less than 2g/day of sodium. Please drink less than 2L of water a day. Call your doctor for weight change greater than five pounds.\nE: Discontinue all medications if feeling well. Please eat a diet containing less than 2g/day of sodium. Please drink less than 2L of water a day. Call your doctor for weight change greater than three pounds.

Your solution should have only 2 parts:  "REASON" and "ANSWER", and start from "REASON".\nThe "REASON" part should be less than 50 words and be your reasoning process about why do you make that choice instead of copy the choice.\nThe "ANSWER" part should be the option letters only. Do not include the medication name or words.\nPlease follow the format strictly.

"REASON": <Your reason here>
"ANSWER": <Your choice (letters only) here>
'''
    # test_content = "Who is Paul Graham"

    if test_function == "msg":
        example_content = test_content
        resp = llm_agent.send_msg(example_content).text
        print("-------------------------------------")
        print(resp)
        print("-------------------------------------")
    
    elif test_function == "sql":
        example_raw_response = os.getenv("example_raw_response")
        example_raw_response = '''
        ```sql
    WITH surgery_procedures AS (
        SELECT d_icd.long_title, COUNT(*) AS procedure_count
        FROM admissions
        JOIN diagnoses_icd ON admissions.subject_id = diagnoses_icd.subject_id
        JOIN d_icd_diagnoses d_icd ON diagnoses_icd.icd_code = d_icd.icd_code
        WHERE admissions.subject_id = 14736532
        AND admission_type = 'Surgical'
        GROUP BY d_icd.long_title
        ORDER BY procedure_count DESC
    )
    SELECT long_title
    FROM surgery_procedures
    LIMIT 2;
    ```
    '''
        example_extracted_codes = llm_agent.extract_code_blocks(example_raw_response, "sql")
        print("codes extracted: " + example_extracted_codes[0])
    

    elif test_function == "extract":
        
        example_raw_response_ehrnoteqa = '''
        "ANSWER": A,
        "REASON": The patient was fairly stabilized, with pain under control, consuming a regular diet, and able to walk and relieve himself without assistance.
        '''
        
        example_raw_response_ehrdsqa = '''
        "REASON": The patient's allergies and adverse drug reactions would be documented in their Electronic Health Record (EHR) under the medication list or allergy section. This information is crucial for healthcare providers to avoid prescribing medications that could potentially harm the patient.
    
        "ANSWER": To determine if the patient has any known allergies or adverse drug reactions, I would need to access the patient's Electronic Health Record (EHR) and review the medication list and allergy section.
    '''
    
        example_raw_response_gemma = '''
    Reason:
    The patient should take aspirin for life to prevent platelet aggregation. The patient should take plavix for 12 months to prevent stent clotting. The patient should not stop plavix without speaking to the cardiologist first. The patient should continue to take all medications as prescribed.
    Answer: B<EOS_TOKEN>
    
    Reason:
    The patient should take aspirin for life to prevent platelet aggregation. The patient should take plavix for 12 months to prevent stent clotting. The patient should not stop plavix without speaking to the cardiologist first. The patient should continue to take all medications as prescribed.
    Answer: B
        '''
        example_answer, example_reason = llm_agent.extract_answer_and_reason(
            example_raw_response_gemma)
        print("ANSWER extracted: " + example_answer)
        print("REASON extracted: " + example_reason)

