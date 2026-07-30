import pandas as pd
import re
import os

def clean_mmlu(text):

    parts = re.split(r'\b[aA]\.\s|\b[aA]\)\s', str(text), maxsplit=1)
    return parts[0].replace("Answer:", "").strip()

def assign_domain_group(subject):

    stem_subjects = {
        'abstract_algebra',
        'anatomy',
        'astronomy',
        'college_biology',
        'college_chemistry',
        'college_computer_science',
        'college_mathematics',
        'college_physics',
        'computer_security',
        'conceptual_physics',
        'electrical_engineering',
        'elementary_mathematics',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_computer_science',
        'high_school_mathematics',
        'high_school_physics',
        'high_school_statistics',
        'machine_learning'}
    return 'STEM' if subject in stem_subjects else 'Non-STEM'

def main():
    print("Data Alignment and Preprocessing")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    questions_path = os.path.join(
        base_dir, "raw_data", "mmlu_questions.csv")
    responses_path = os.path.join(
        base_dir, "raw_data", "mmlu_model_responses.csv")

    out_dir = os.path.join(base_dir, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mmlu_aligned.csv")

    df_q = pd.read_csv(questions_path)
    df_r = pd.read_csv(responses_path)

    print(f"\nLoaded Questions Shape: {df_q.shape}")
    print(f"Loaded Responses Shape: {df_r.shape}")

    model_col = df_r.columns[0]
    model_names = df_r[model_col].tolist()
    item_cols = list(df_r.columns)[1:]

    print(f"Number of Models: {len(model_names)}")
    print(f"Number of Items in Response Matrix: {len(item_cols)}")

    print("\n\nCleaning Question Text and Assigning a Priori Domain Groups")
    df_q['clean_text'] = df_q['question_text'].apply(clean_mmlu)
    df_q['domain_group'] = df_q['subject'].apply(assign_domain_group)

    subject_groups = {sub: group for sub, group in df_q.groupby('subject')}

    aligned_rows = []

    for col in item_cols:
        try:
            parts = col.replace('mmlu_', '').split('_row_')
            subject = parts[0]
            row_index = int(parts[1])

            q_row = subject_groups[subject].iloc[row_index]

            item_record = {
                'question_id': col,
                'item_id': q_row['item_id'],
                'subject': q_row['subject'],
                'question_text': q_row['question_text'],
                'clean_text': q_row['clean_text'],
                'choices': q_row['choices'],
                'ground_truth': q_row['ground_truth'],
                'domain_group': q_row['domain_group']
            }

            responses_for_item = df_r[col].tolist()
            for model_index, model_name in enumerate(model_names):
                item_record[model_name] = responses_for_item[model_index]

            aligned_rows.append(item_record)

        except Exception as e:
            print(f"Failed to Align Column {col}. Reason: {e}")

    df_data = pd.DataFrame(aligned_rows)
    print(f"\nShape of the Final Dataset: {df_data.shape}")

    stem_count = df_data[df_data['domain_group'] == 'STEM'].shape[0]
    nonstem_count = df_data[df_data['domain_group'] == 'Non-STEM'].shape[0]
    print(f"STEM Items: {stem_count} ({stem_count / len(df_data) * 100:.2f}%)")
    print(
        f"Non-STEM Items: {nonstem_count} ({nonstem_count / len(df_data) * 100:.2f}%)")

    df_data.to_csv(out_path, index=False)
    print("\nAlignment Completed, Final Dataset Saved to", out_path)

if __name__ == '__main__':
    main()
