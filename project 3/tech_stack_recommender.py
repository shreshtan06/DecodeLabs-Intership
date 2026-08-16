import csv
import math
import os
import re
from collections import Counter

# STEP 0: Shared vocabulary helper
def tokenize(text):
    """Lowercase and split text into word tokens (shared vocabulary space)."""
    return re.findall(r"[a-z0-9]+", text.lower())

# STEP 1 (part A): INGESTION - load the item dataset (job roles = "items")
def load_dataset(csv_path):
       if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    items = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            role = (row.get("job_role") or "").strip()
            skills = row.get("skills") or ""
            if not role:
                continue
            items.append({"role": role, "tokens": tokenize(skills)})

    if not items:
        raise ValueError("Dataset is empty - no job roles were loaded.")
    return items
# STEP 1 (part B): INGESTION - capture user state via real console input
def get_user_skills(min_inputs=3):
    
    print(f"Enter at least {min_inputs} skills or interests (comma-separated).")
    print('Example: Python, Cloud Computing, Automation')

    while True:
        raw = input("Your skills: ").strip()
        skills = [s.strip() for s in raw.split(",") if s.strip()]

        if len(skills) >= min_inputs:
            return skills

        print(f"  -> Need at least {min_inputs} skills, "
              f"you entered {len(skills)}. Please try again.\n")


def compute_idf(all_token_lists):
    """
    IDF(term) = log(Total Documents / Documents containing term)
    Computed once across the whole corpus (all job roles + user profile).
    """
    n_docs = len(all_token_lists)
    doc_freq = Counter()
    for tokens in all_token_lists:
        for term in set(tokens):
            doc_freq[term] += 1

    return {
        term: math.log(n_docs / df) if df > 0 else 0.0
        for term, df in doc_freq.items()
    }


def tfidf_vector(tokens, idf):
    """
    TF(term) = count(term in doc) / total terms in doc
    weight    = TF * IDF
    Returns a sparse vector: {term: weight}
    """
    if not tokens:
        return {}
    total_terms = len(tokens)
    tf = Counter(tokens)
    return {
        term: (count / total_terms) * idf.get(term, 0.0)
        for term, count in tf.items()
    }

def cosine_similarity(vec_a, vec_b):
    """
    cos(theta) = (A . B) / (||A|| * ||B||)
    Operates on sparse dict vectors - no dense matrix required.
    """
    shared_terms = set(vec_a) & set(vec_b)
    dot = sum(vec_a[t] * vec_b[t] for t in shared_terms)

    mag_a = math.sqrt(sum(w * w for w in vec_a.values()))
    mag_b = math.sqrt(sum(w * w for w in vec_b.values()))

    if mag_a == 0 or mag_b == 0:
        return 0.0

    return dot / (mag_a * mag_b)


# STEPS 2-4 combined: the recommendation engine
def recommend(user_skills, items, top_n=3, min_inputs=3):
    """
    user_skills : list[str]  e.g. ["Python", "Cloud Computing", "Automation"]
    items       : output of load_dataset()
    top_n       : how many results to return (Filtering step)
    min_inputs  : minimum number of skills required (data density guard)

    Returns a list of (role, score) tuples sorted best-first.
    Returns an empty list if a Cold Start is detected.
    """
    if len(user_skills) < min_inputs:
        raise ValueError(
            f"At least {min_inputs} user inputs are required "
            f"for sufficient data density (got {len(user_skills)})."
        )

    # Build shared vocabulary / IDF across item corpus + user profile
    all_docs = [item["tokens"] for item in items]
    user_tokens = tokenize(" ".join(user_skills))
    idf = compute_idf(all_docs + [user_tokens])

    # Vectorize the user profile (Input / User State)
    user_vector = tfidf_vector(user_tokens, idf)

    # STEP 2: Scoring - compare user vector to every item vector
    scored = []
    for item in items:
        item_vector = tfidf_vector(item["tokens"], idf)
        score = cosine_similarity(user_vector, item_vector)
        scored.append((item["role"], score))

    # STEP 3: Sorting - descending by similarity score
    scored.sort(key=lambda pair: pair[1], reverse=True)

    # Cold Start check: if the best score is 0, the user's skills share
    # zero vocabulary with the dataset - nothing meaningful to recommend.
    if not scored or scored[0][1] == 0.0:
        return []

    # STEP 4: Filtering - Top-N cutoff to prevent choice overload
    return scored[:top_n]

# OUTPUT: Display recommended items
def print_recommendations(user_skills, results, top_n=3):
    print("\n" + "=" * 50)
    print(f"Your input skills : {', '.join(user_skills)}")
    print("=" * 50)

    if not results:
        print("\n⚠  Cold Start detected — none of your skills match our "
              "current vocabulary.")
        print("   Falling back to trending / globally popular roles is "
              "recommended.\n")
        return

    print(f"\nTop {len(results)} recommended career paths:\n")
    for rank, (role, score) in enumerate(results, start=1):
        pct = score * 100
        bar = "█" * int(pct / 5)
        print(f"  {rank}. {role:<28} {pct:5.1f}%  {bar}")
    print()

def main():
    dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "raw_skills.csv")

    print(" --------Tech Stack Recommender--------")
    items = load_dataset(dataset_path)
    user_skills = get_user_skills(min_inputs=3)
    results = recommend(user_skills, items, top_n=3, min_inputs=3)
    print_recommendations(user_skills, results, top_n=3)
if __name__ == "__main__":
    main()
