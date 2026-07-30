import os
import warnings

import pandas as pd
import numpy as np
import jax
from jax import random
import numpyro
import numpyro.distributions as dist
from numpyro.infer import SVI, Trace_ELBO, autoguide

from config import RESULTS_DIR, resolve

os.makedirs(RESULTS_DIR, exist_ok=True)

def irt_2pl_model(responses):
    num_models, num_items = responses.shape

    theta = numpyro.sample("theta", dist.Normal(0.0, 1.0).expand([num_models]))

    b = numpyro.sample("b", dist.Normal(0.0, 1.0).expand([num_items]))

    a = numpyro.sample("a", dist.HalfNormal(1.0).expand([num_items]))

    logits = a * (theta[:, None] - b)

    with numpyro.plate("models", num_models, dim=-2):
        with numpyro.plate("items", num_items, dim=-1):
            numpyro.sample("obs", dist.Bernoulli(logits=logits), obs=responses)

def main():
    warnings.simplefilter(
        action='ignore', category=pd.errors.PerformanceWarning)

    devices = jax.devices()
    backend = devices[0].platform.upper()

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    in_path = resolve("mmlu_aligned.csv")
    out_dir = RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "mmlu_IRT_calibrated.csv")
    abilities_path = os.path.join(out_dir, "mmlu_model_abilities.csv")
    cached_out = resolve("mmlu_IRT_calibrated.csv")
    cached_abilities = resolve("mmlu_model_abilities.csv")

    results_path = os.path.join(out_dir, "results_summary.txt")

    df_data = pd.read_csv(in_path)

    for col in ['difficulty_score', 'discrimination_score',
                'difficulty_score.1', 'discrimination_score.1']:
        if col in df_data.columns:
            df_data = df_data.drop(columns=[col])

    question_details = [
        'question_id', 'item_id', 'subject', 'question_text',
        'clean_text', 'choices', 'ground_truth', 'domain_group'
    ]
    model_cols = [
        col for col in df_data.columns if col not in question_details]

    responses = df_data[model_cols].values.T.astype(float)
    num_models, num_items = responses.shape

    cache_loaded = False
    if os.path.exists(cached_out) and os.path.exists(cached_abilities):
        try:
            df_out = pd.read_csv(cached_out)
            df_ab = pd.read_csv(cached_abilities)
            if len(df_out) == len(df_data) and 'difficulty_score' in df_out.columns and 'discrimination_score' in df_out.columns:
                b_est = df_out['difficulty_score'].values
                a_est = df_out['discrimination_score'].values
                ab_dict = dict(zip(df_ab['model_name'], df_ab['theta_score']))
                theta_est = np.array([ab_dict[m] for m in model_cols])
                print("Found fully calibrated IRT parameters in output cache. Skipping SVI optimization.")
                cache_loaded = True
        except Exception:
            pass

    if not cache_loaded:

        print("Initiating IRT Calibration using NumPyro SVI")
        guide = autoguide.AutoDiagonalNormal(irt_2pl_model)
        optimizer = numpyro.optim.Adam(step_size=0.01)
        svi = SVI(irt_2pl_model, guide, optimizer, loss=Trace_ELBO())
        svi_result = svi.run(random.PRNGKey(
            1729), num_steps=5000, responses=responses, progress_bar=True)
        params = svi_result.params

        medians = guide.median(params)
        theta_est = np.array(medians['theta'])
        b_est = np.array(medians['b'])
        a_est = np.array(medians['a'])

        df_abilities = pd.DataFrame({
            'model_name': model_cols,
            'theta_score': theta_est
        }).sort_values(by='theta_score', ascending=False)
        df_abilities.to_csv(abilities_path, index=False)

    df_data['difficulty_score'] = b_est
    df_data['discrimination_score'] = a_est

    cols_order = question_details + \
        ['difficulty_score', 'discrimination_score'] + model_cols
    df_data = df_data[cols_order]
    df_data.to_csv(out_path, index=False)

    stem_diff = df_data[df_data['domain_group'] == 'STEM']['difficulty_score']
    nonstem_diff = df_data[df_data['domain_group']
                           == 'Non-STEM']['difficulty_score']

    with open(results_path, "w", encoding="utf-8") as f:
        f.write("RESULTS SUMMARY\n\n")

        f.write("IRT CALIBRATION VALIDITY STATISTICS\n\n")
        f.write(f"  Total Models           : {num_models}\n")
        f.write(f"  Total Items            : {num_items}\n\n")

        f.write(" Item Difficulty (b-parameter)\n")
        f.write(f"    Mean: {b_est.mean():>6.4f} | Std: {b_est.std():>6.4f}\n")
        f.write(
            f"    Min:  {b_est.min():>6.4f} | Max: {b_est.max():>6.4f}\n\n")

        f.write(" Item Discrimination (a-parameter)\n")
        f.write(f"    Mean: {a_est.mean():>6.4f} | Std: {a_est.std():>6.4f}\n")
        f.write(
            f"    Min:  {a_est.min():>6.4f} | Max: {a_est.max():>6.4f}\n\n")

        f.write(" Model Abilities (theta-parameter)\n")
        f.write(
            f"    Mean: {theta_est.mean():>6.4f} | Std: {theta_est.std():>6.4f}\n")
        f.write(
            f"    Min:  {theta_est.min():>6.4f} | Max: {theta_est.max():>6.4f}\n\n")

        f.write(" Domain-Stratified Difficulty\n")
        f.write(
            f"    STEM     Mean: {stem_diff.mean():>6.4f} | Std: {stem_diff.std():>6.4f}\n")
        f.write(
            f"    Non-STEM Mean: {nonstem_diff.mean():>6.4f} | Std: {nonstem_diff.std():>6.4f}\n\n")

    print(f"IRT Calibration completed. Results saved to {results_path}")

if __name__ == '__main__':
    main()
