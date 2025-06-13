from EHRAGent import EHRAGent
from utils.eval_utils import Evaluator
from utils.record_utils import ResultProcessor
import os
from utils.args_utils import parse_arguments, get_config_or_args

dir_now = os.path.dirname(os.path.abspath(__file__))
config_dir = dir_now + "/../config/benchmark_experiments/"

# Usage for default config:
# 1. rag: --config "default" --is_rag True --dataset "benchmark_medications"
# 2. direct ask: --config "default" --dataset "benchmark_medications"


def main():
    # Parse command-line arguments
    args = parse_arguments()

    keyword = "medications" if args.dataset == "benchmark_medications" else "diagnosis"
    # If no config is provided via command-line, use the default config file
    if args.config=="default":
        if not args.is_rag:
            args.config = config_dir + f"config_{keyword}_direct_ask_without_reasoning.yaml"
        else:
            args.config = config_dir + f"config_{keyword}_rag_{args.rag_option_similar_patients}_answers.yaml"
        print(f"No customized config file provided, using default: {args.config}")

    # Load configuration from YAML or individual inputs
    config = get_config_or_args(args)
    # Access configuration values
    notes = config['notes']
    llm = config['llm']
    dataset = config['dataset']
    num_questions = config['num_questions']

    qa_file_path = dir_now + config['qa_file']
    prompt_file_path = dir_now + config['prompt_file']
    database_file_path = dir_now + config['database_file']

    is_rag = config['is_rag']

    # Print info
    print(f"Notes: {notes}")

    print(f"LLM: {llm}")
    print(f"Dataset: {dataset}")
    print(f"Number of Questions: {num_questions}")

    print(f"QA File: {qa_file_path}")
    print(f"Prompt File Path: {prompt_file_path}")
    print(f"Database File Path: {database_file_path}")

    print(f"Is rag?: {is_rag}")

    if is_rag:
        rag_tool = config['rag_tool']
        rag_query_details_level = config['rag_query_details_level']
        print(f"RAG query details level: {rag_query_details_level}")
        print(f"Separate or combine info from similar_patients: {args.rag_option_similar_patients}")

        print(f"rag tool is {rag_tool}")
        similar_patients_map_file = dir_now + config['similar_patients_map_file']
        print(f"similar_patients_map_file: {similar_patients_map_file}")

        if rag_query_details_level == 2: # "summarized background" = "Brief Hospital Course" + "Pertinent Results"
            tmp_prompt_for_similar_patients_file = dir_now + config['tmp_prompt_rag_query_with_similar_patients_background_file']
        else: # 0 as default question only; 1 TODO only "Brief Hospital Course"
            tmp_prompt_for_similar_patients_file = dir_now + config['tmp_prompt_for_similar_patients_file']
        print(f"tmp_prompt_for_similar_patients_file: {tmp_prompt_for_similar_patients_file}")


    ehr_rag_agent = EHRAGent(llm=llm)

    # ----------------Settings-------------
    ehr_rag_agent.set_max_iteration(num_questions)
    ehr_rag_agent.set_dataset(dataset)
    ehr_rag_agent.set_rag_info_from_similar_patients = args.rag_option_similar_patients
    ehr_rag_agent.embedding_scan_ranker = False  # configurable as False for loading similar map; True to scan embeddings for each question everytime
    # ----------------Settings-------------

    ehr_rag_agent.load_questions(qa_file_path, prompt_file_path, None)
    ehr_rag_agent.prompt_loader.set_max_qa(ehr_rag_agent.max_iteration)
    correct_answer_dict = ehr_rag_agent.prompt_loader.get_info_dict("correct_answer")
    patient_id_dict = ehr_rag_agent.prompt_loader.get_info_dict("hadm_id")
    question_dict = ehr_rag_agent.prompt_loader.get_info_dict("question")
    if dataset == "benchmark_diagnosis":
        options_length_dict = ehr_rag_agent.prompt_loader.get_info_dict("discharge_diagnosis_options", is_length=True)
    else:
        options_length_dict = ehr_rag_agent.prompt_loader.get_info_dict("discharge_medications_options", is_length=True)


    # Final result
    if not is_rag:
        # No RAG
        responses_dict, summary_retrieved_dict, _ = ehr_rag_agent.process_all_prompts(qa_file=qa_file_path,
                                                                                      is_rag_tool=False,
                                                                                      one_time_ask=True
                                                                                      )
    else:
        # With RAG
        ehr_rag_agent.tmp_prompt_for_similar_patients = tmp_prompt_for_similar_patients_file
        responses_dict, summary_retrieved_dict, _ = ehr_rag_agent.process_all_prompts(qa_file=qa_file_path,
                                                                                      is_rag_tool=True,
                                                                                      similarity_map_path=similar_patients_map_file,
                                                                                      rag_tool=rag_tool)
    # Process result
    choice_dict, choice_reasons_dict = ehr_rag_agent.extract_letter_from_result_dict(responses_dict, is_choice_only=True)

    print("Start evaluation")
    evaluator = Evaluator(dataset=dataset)
    evaluator.set_correct_answers(correct_answer_dict)
    evaluator.set_user_answers(choice_dict)
    evaluator.set_total_choices(options_length_dict)
    total_correctness_rate, correctness_scores_dict, is_correct_dict, f_score_dict, f_score_avg= evaluator.evaluate_benchmark()
    print("Finished evaluation")

    print("Start result processor")
    llm_name = llm.replace("/", "_")
    result = f"correct_rate_{total_correctness_rate}_f_score_avg_{f_score_avg}"
    result_processor = ResultProcessor(f"{result}_run_{num_questions}_{dataset}_"+llm_name+"_"+notes+"_",
                                       is_long_name=False,
                                       directory=f"../data/result_csv/{dataset}/")
    result_processor.add_multiple_data(hadm_id=patient_id_dict,
                                       correctness_scores=correctness_scores_dict,
                                       f_score_dict=f_score_dict,
                                       Is_correct=is_correct_dict,
                                       Correct=correct_answer_dict,
                                       Chosen=choice_dict,
                                       Options_length_dict=options_length_dict,
                                       Choice_reason=choice_reasons_dict,
                                       Question=question_dict,
                                       Response=responses_dict,
                                       Retrived_knowledge=summary_retrieved_dict)
    result_processor.generate_csv()
    print("Total Correctness_rate is: ")
    print(total_correctness_rate)
    print("Average f-score is: ")
    print(f_score_avg)

if __name__ == '__main__':
    main()