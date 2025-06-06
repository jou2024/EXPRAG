# Load components (Knowledge, tools, system_prompt), output a completed prompt

import json


class QAPromptLoader:
    def __init__(self, qa_file=None, template_file=None, knowledge_file=None):
        self.qa_file = qa_file
        self.template_file = template_file
        self.knowledge_file = knowledge_file
        self.questions_answers = self.load_qa() if qa_file else None
        self.template = self.load_template() if template_file else None
        self.knowledge = self.load_knowledge() if knowledge_file else ""
        self.current_id_index = 0
        self.max_qa_pairs = len(self.questions_answers) if self.questions_answers else 0

    def set_max_qa(self, max_qa_pairs):
        self.max_qa_pairs = max_qa_pairs

    def load_qa(self):
        """Load questions and answers from the JSON file."""
        with open(self.qa_file, 'r') as file:
            return json.load(file)

    def load_template(self):
        """Load the prompt template from an external file, if provided."""
        if self.template_file:
            with open(self.template_file, 'r') as file:
                return file.read()
        return None

    def load_knowledge(self):
        """Load additional knowledge content from a specified file."""
        if self.knowledge_file:
            with open(self.knowledge_file, 'r') as file:
                return file.read()
        return ""

    def replace_place_holder(self, prompt, placeholder, value):
        updated_prompt = prompt.replace(f"{{{placeholder}}}", str(value))
        return updated_prompt

    def next_prompt(self, extra_placeholders=None, update_index=True, given_qa=None):
        """Generate the next prompt based on the current ID index, replacing placeholders.

        Args:
            extra_placeholders (dict, optional): Additional placeholders and their corresponding
                                                 keys in the QA dictionary to replace in the template.
                                                 Example: {'patient_id': 'patient_id'}
                                                 This will replace '{patient_id}' with the value of 'patient_id' from qa
            update_index (bool, optional): Default as true for one prompt questions
            given_qa (dict, optional): Single QA input dictionary including at least "question"
        Returns:
            str: The prompt with replaced placeholders.
        """
        if self.current_id_index >= self.max_qa_pairs:
            print("No more Q&A pairs.")
            return None

        qa = self.questions_answers[self.current_id_index] if given_qa is None else given_qa
        if update_index:
            self.current_id_index += 1  # Move to the next Q&A pair for the next call

        # Prepare the prompt by replacing the basic placeholders
        prompt = self.template if self.template else "If you have knowledge about EHR {knowledge} .Question: {question}"
        prompt = prompt.replace("{question}", qa.get("question", "")).replace("{knowledge}", self.knowledge)

        # Replace extra placeholders if any are specified
        if extra_placeholders:
            for placeholder, key in extra_placeholders.items():
                value = qa.get(key, '')
                prompt = self.replace_place_holder(prompt, placeholder, value)

        return prompt

    def get_info_dict(self, info, is_length=False):
        # "answer" "patient_id" "question"
        info_dict = {}
        index = 0
        while index < self.max_qa_pairs:
            info_value = self.questions_answers[index].get(info, "")
            if not is_length:
                info_dict[index] = info_value
            else:
                info_dict[index] = len(info_value)
            index += 1
        return info_dict


# Main function
if __name__ == "__main__":

    from dotenv import load_dotenv
    import os
    load_dotenv()  # Load environment variables from a .env file

    example_qa_file = os.getenv('QA_FILE_PATH')  # The path to the JSON file
    example_template_file = os.getenv('TEMPLATE_FILE_PATH')  # Path to the template file (optional)
    example_knowledge_file = os.getenv('KNOWLEDGE_FILE_PATH')  # Path to the knowledge file (optional)

    prompt_loader = QAPromptLoader(example_qa_file, example_template_file, example_knowledge_file)

    print("Number of Questions: ")
    print(len(prompt_loader.questions_answers))

    print("Dictionary of answers")
    print(prompt_loader.get_info_dict("answer"))

    prompt_loader.set_max_qa(3)  # Get the number of entries in questions_answers
    print("Run max number of questions: ")
    print(prompt_loader.max_qa_pairs)

    while prompt_loader.current_id_index < prompt_loader.max_qa_pairs:
        example_prompt = prompt_loader.next_prompt(
            {'patient_id': 'patient_id',
             'choice_A': 'choice_A',
             'choice_B': 'choice_B',
             'choice_C': 'choice_C',
             'choice_D': 'choice_D',
             'choice_E': 'choice_E'}
        )
        if example_prompt is None:
            print("No more prompts available.")
            break
        else:
            print(" \n \n Prompt: \n", example_prompt)

    # # Example: Generate the first prompt
    # example_prompt = prompt_loader.next_prompt(
    #     {'patient_id': 'patient_id',
    #      'choice_A': 'choice_A',
    #      'choice_B': 'choice_B',
    #      'choice_C': 'choice_C',
    #      'choice_D': 'choice_D',
    #      'choice_E': 'choice_E'}
    # )
    # if example_prompt:
    #     print(example_prompt)



