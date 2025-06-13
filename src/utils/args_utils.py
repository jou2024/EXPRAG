import yaml
import argparse

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run RAG experiments")

    # Accept config path or individual inputs
    parser.add_argument("--config", type=str, help="Path to the YAML config file")
    parser.add_argument("--llm", type=str, help="LLM model (e.g., '4o')")
    parser.add_argument("--dataset", type=str, help="Dataset name (e.g., 'benchmark_medications')")
    parser.add_argument("--num_questions", type=int, help="Number of questions to ask")
    parser.add_argument("--notes", type=str, default="", help="Experiment notes")
    parser.add_argument("--is_rag", type=bool, default=False, help="Is RAG mode enabled")
    parser.add_argument("--rag_tool", type=str, default="auto_merging", help="rag_tool or retriever model")

    # File paths
    parser.add_argument("--qa_file", type=str, help="Path to the QA file")
    parser.add_argument("--prompt_file", type=str, help="Path to the prompt file")
    parser.add_argument("--database_file", type=str, help="Path to the database file")

    parser.add_argument("--similar_patients_map_file", type=str, help="Path to the similar patients map file")

    parser.add_argument("--rag_option_similar_patients", type=str, default="combine", help="Combine all the similar patients files, or do RAG for all separate files")
    parser.add_argument("--rag_query_details_level", type=int, default=0, help="Number as level of the rag query similar patients: 0 as default only question, 1 for background summary")

    parser.add_argument("--tmp_prompt_for_similar_patients_file", type=str, help="Path to the tmp_prompt_for_similar_patients_file")
    parser.add_argument("--tmp_prompt_rag_query_with_similar_patients_background_file", type=str,
                        help="Path to the tmp_prompt_rag_query_with_similar_patients_background_file")

    return parser.parse_args()

def get_config_or_args(args):
    """
    Load from config if provided, else use individual command-line inputs.
    Returns a dictionary of configuration values.
    """
    if args.config:
        config = load_config(args.config)
        return {
            "llm": config.get("llm", args.llm),
            "dataset": config.get("dataset", args.dataset),
            "num_questions": config.get("num_questions", args.num_questions),
            "notes": config.get("notes", args.notes),
            "is_rag": config.get("is_rag", args.is_rag),
            "rag_tool": config.get("rag_tool", args.rag_tool),
            "rag_query_details_level": config.get("rag_query_details_level", args.rag_query_details_level),
            "qa_file": config["paths"].get("qa_file"),
            "prompt_file": config["paths"].get("prompt_file"),
            "database_file": config["paths"].get("database_file"),
            "similar_patients_map_file": config["paths"].get("similar_patients_map_file"),
            "tmp_prompt_for_similar_patients_file": config["paths"].get("tmp_prompt_for_similar_patients_file"),
            "tmp_prompt_rag_query_with_similar_patients_background_file": config["paths"].get("tmp_prompt_rag_query_with_similar_patients_background_file"),
        }
    else:
        return {
            "llm": args.llm,
            "dataset": args.dataset,
            "num_questions": args.num_questions,
            "notes": args.notes,
            "is_rag": args.is_rag,
            "rag_tool": args.rag_tool,
            "qa_file": args.qa_file,
            "prompt_file": args.prompt_file,
            "database_file": args.database_file,
            "rag_query_details_level": args.rag_query_details_level,
            "similar_patients_map_file": args.similar_patients_map_file,
            "tmp_prompt_for_similar_patients_file": args.tmp_prompt_for_similar_patients_file,
            "tmp_prompt_rag_query_with_similar_patients_background_file": args.tmp_prompt_rag_query_with_similar_patients_background_file,

        }
