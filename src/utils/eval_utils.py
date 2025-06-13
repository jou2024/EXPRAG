import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from llm_utils import LLMAgent


class Evaluator:
    def __init__(self, dataset="EHRNoteQA"):
        self.correct_answers = {}  # Stores the correct answers, e.g., {'1': 'A', '2': 'B'}
        self.user_answers = {}  # Stores the user's answers, e.g., {'1': 'A', '2': 'C'}
        self.dataset = dataset
        self.total_choices = {}
        self.llm_agent = None
        if dataset == "EHRNoteQA":
            self.llm_agent = LLMAgent(llm="4o")

    def set_correct_answers(self, correct_answers):
        """Sets the correct answers from a dictionary."""
        self.correct_answers = correct_answers

    def set_user_answers(self, user_answers):
        """Sets the user's answers from a dictionary."""
        self.user_answers = user_answers

    def calculate_correctness_rate(self):
        """Calculates the correctness rate based on the user's answers and the correct answers."""
        if not self.correct_answers:
            raise ValueError("Correct answers must be set before evaluating.")

        total_questions = len(self.correct_answers)
        if total_questions == 0:
            return 0  # Avoid division by zero if no questions are provided

        correct_count = sum(1 for q_id, ans in self.user_answers.items() if self.correct_answers.get(q_id) == ans)
        correctness_rate = (correct_count / total_questions) * 100
        return correctness_rate

    def evaluate(self):
        """Evaluates the answers and returns the correctness rate and a dictionary of correct flags."""
        try:
            correct_flags = {}
            correct_count = 0

            for q_id, correct_ans in self.correct_answers.items():
                user_ans = self.user_answers.get(q_id)
                if self.eval_compare(user_ans, correct_ans):
                    correct_flags[q_id] = 1
                    correct_count += 1
                else:
                    correct_flags[q_id] = 0

            total_correctness_rate = self.cal_total_correctness_rate(correct_count)

            return total_correctness_rate, correct_flags

        except ValueError as e:
            print(e)
            return None

    def set_total_choices(self, total_choices):
        """Sets the total number of options for each question."""
        self.total_choices = total_choices

    def calculate_average(self, dictionary):
        return sum(dictionary.values()) / len(dictionary) if dictionary else 0

    def cal_total_correctness_rate(self, correct_count):
        total_questions = len(self.correct_answers)
        if total_questions > 0:
            total_correctness_rate = (correct_count / total_questions) * 100
            total_correctness_rate = round(total_correctness_rate, 2)
        else:
            total_correctness_rate = 0  # Avoid division by zero if no questions are provided

        return total_correctness_rate

    def evaluate_benchmark(self):
        """Evaluates the correctness for both 'Benchmark_medication' and 'Benchmark_diagnosis' datasets."""
        correct_flags = {}
        correctness_scores = {}
        f_score_dict = {}

        correct_count = 0
        for q_id, correct_ans in self.correct_answers.items():
            user_ans = self.user_answers[q_id]
            total_choice_count = self.total_choices[q_id]

            if not user_ans:
                f_score = 0
                f_score_dict[q_id] = f_score
                print(f" !! id {q_id} user_ans is invalid")
                continue
            if total_choice_count == 0:
                correctness_scores[q_id] = 0
                print(" No total counts")
                continue

            # Convert answers to sets for comparison
            correct_ans_set = set(correct_ans)
            user_ans_set = set(user_ans)

            # Calculate correctness
            true_positives = len(correct_ans_set & user_ans_set)  # Correctly chosen options
            false_negatives = len(correct_ans_set - user_ans_set)  # Correct options not chosen
            false_positives = len(user_ans_set - correct_ans_set)  # Incorrect options chosen

            correct_total = true_positives + (total_choice_count - len(correct_ans_set) - false_positives)

            # Calculate precision, recall, and F-score
            if true_positives + false_positives > 0:
                precision = true_positives / (true_positives + false_positives)
            else:
                precision = 0

            if true_positives + false_negatives > 0:
                recall = true_positives / (true_positives + false_negatives)
            else:
                recall = 0

            if precision + recall > 0:
                f_score = round(2 * (precision * recall) / (precision + recall), 3)
            else:
                f_score = 0

            f_score_dict[q_id] = f_score
            # You can print or return the results
            # print(f"    Precision: {precision}")
            # print(f"    Recall: {recall}")
            # print(f"    F-score: {f_score}")

            if correct_total < 0:
                correctness_scores[q_id] = 0
                correct_flags[q_id] = -1
                print(f" !!! {q_id}Negative correct answer!  Invalid user answer")
                continue

            correctness_scores[q_id] = round(correct_total / total_choice_count, 3)
            correct_flags[q_id] = 1 if correctness_scores[q_id] == 1.0 else 0

            if correct_flags[q_id] == 1:
                correct_count += 1

        total_correctness_rate = self.cal_total_correctness_rate(correct_count)
        f_score_avg = round(self.calculate_average(f_score_dict), 3)
        return total_correctness_rate, correctness_scores, correct_flags, f_score_dict, f_score_avg

    def eval_compare(self, user_ans, correct_ans):
        if self.dataset in ["EHRNoteQA", "benchmark_choice"]:
            return user_ans == correct_ans
        elif self.dataset in ["benchmark_medication", "benchmark_diagnosis"]:
            return self.eval_by_llm(user_ans, correct_ans)
        else:
            return False

    def eval_by_llm(self, user_ans, correct_ans):
        prompt_eval = self.prompt_eval(user_ans, correct_ans)
        llm_eval_answer = self.llm_agent.send_msg(prompt_eval)
        answer, reason = self.llm_agent.extract_answer_and_reason(llm_eval_answer.text, self.dataset)
        if answer == 'True' or ("<True>" in llm_eval_answer.text):
            return True
        elif answer == 'False' or ("<False>" in llm_eval_answer.text):
            return False
        else:
            print("!!!!!!!!!!!")
            print(f"Answer as Evaluation result is {llm_eval_answer.text}")
            print(f"User Answer is {user_ans}")
            print(f"Correct Answer is {correct_ans}")
            return False

    def prompt_eval(self, user_ans, correct_ans):
        prompt = f"""
        You are a doctor and an expert in Electronic Health Records (EHR).

        Your task is to determine if an answer provided by another doctor, based on reasoning from similar patients, is correct.

        The correct answer is: {correct_ans}
        The answer from the user is: {user_ans}

        Does the user's answer convey the same meaning as the correct answer?

        Please respond in the following format:
        "ANSWER": <True or False>
        """

        return prompt


# Example Usage
if __name__ == "__main__":
    example_correct = {'1': 'A', '2': 'B', '3': 'C', '4': 'D'}
    example_user = {'1': 'A', '2': 'C', '3': 'C', '4': 'D'}

    example_eval = Evaluator()
    example_eval.set_correct_answers(example_correct)
    example_eval.set_user_answers(example_user)
    example_correctness_rate, example_correct_flags = example_eval.evaluate()
    print("Correctness rate is: ")
    print(example_correctness_rate)


    # Example correct answers where 'ABCDE' are the correct options
    example_correct = {'1': 'ABCDE', '2': 'BC'}

    # Example user answers: User chooses 'ABCDH' and 'B'
    example_user = {'1': 'ABCDH', '2': 'B'}

    # Example total options: 8 options for question 1, 4 options for question 2
    example_total_choices = {'1': 8, '2': 4}

    # Initialize evaluator
    example_eval = Evaluator(dataset="Benchmark_medication")

    # Set the correct answers, user answers, and total choices
    example_eval.set_correct_answers(example_correct)
    example_eval.set_user_answers(example_user)
    example_eval.set_total_choices(example_total_choices)

    # Evaluate and print the correctness scores and flags
    _, example2_correctness_scores, example2_correct_flags, example2_f_score_dict, example2_f_score_avg = example_eval.evaluate_benchmark()

    print("Correctness Scores: ", example2_correctness_scores)
    print("Correct Flags: ", example2_correct_flags)
    print("F-score: ", example2_f_score_dict)
    print("Avg F-score: ", example2_f_score_avg)