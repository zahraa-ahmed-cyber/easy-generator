## Synthetic Review Generator – EdTech Course Reviews

This project generates a synthetic dataset of **online course reviews** in the EdTech domain, with **strong quality guardrails** and **multi‑model support** (Groq + Cerebras). It was built as an answer to the “Synthetic Data Generator with Quality Guardrails” assignment.

---

## 1. Project Overview

- **Domain**: EdTech / online learning platform course reviews (e.g., Coursera/Udemy‑style platforms)
- **Goal**: Generate 300–500 realistic, diverse reviews with:
  - Configurable personas, rating distribution, and review styles
  - Multiple LLM providers (Groq + Cerebras)
  - Quality guardrails (length, domain realism, sentiment–rating alignment, diversity, bias)
  - Comparison against real Coursera reviews

Main components:
- `app.py` – end‑to‑end CLI pipeline
- `config/config.yaml` – global config (models, distribution, guardrails, paths)
- `config/personas.yaml` – personas and their characteristics
- `config/prompts.yaml` – prompt templates, anti‑patterns, sentiment guidelines
- `data/domain_keywords.txt` – domain vocabulary for realism checks
- `data/real_reviews.jsonl` – ~48 real reviews for comparison
- `output/` – synthetic dataset, rejected samples, metrics, logs, quality report

---

 ## 1.1. Why Course Reviews?

This project focuses specifically on **EdTech course reviews**—a domain directly relevant to companies like [Easygenerator](https://www.easygenerator.com/en/) that create and manage e-learning content. This strategic choice demonstrates:

### Business Alignment
- **Domain Understanding**: Course reviews are central to understanding learner feedback, course quality, and user satisfaction in the e-learning space
- **Practical Relevance**: Synthetic review data can help e-learning platforms:
  - Analyze learner feedback patterns at scale
  - Test and improve recommendation systems
  - Train sentiment analysis models for course quality assessment
  - Generate training data for A/B testing course improvements
  - Understand review distribution and bias in user feedback

### Technical Relevance
- **AI Integration**: Easygenerator's EasyAI feature uses AI for course creation—this project demonstrates LLM integration and quality control for AI-generated content
- **Quality Guardrails**: The multi-layered validation approach mirrors what's needed for reliable AI-generated content in production
- **Scalability**: The synthetic data generation approach can scale to support large-scale analysis and testing

### Strategic Value
This domain choice shows:
- **Research & Understanding**: Deep knowledge of the e-learning industry and its challenges
- **Business Acumen**: Ability to connect technical solutions to real business problems
- **Strategic Thinking**: Choosing a problem space that's both technically interesting and directly relevant to the target company's mission

---

## 2. Setup

### 2.1. Environment

```bash
python -m venv .venv
source .venv/bin/activate  
pip install -r requirements.txt
```

### 2.2. API Keys (`.env`)

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_key_here
CEREBRAS_API_KEY=your_cerebras_key_here
```

### 2.3. Real Reviews (Already Included)

- `data/real_reviews.jsonl` – contains real Coursera reviews (approx. 48).
- Loaded automatically by `load_real_reviews()`.

---

## 3. How to Run

From the project root:

```bash
python app.py
```

The pipeline (`SyntheticReviewGenerator.run()`) does:
1. **Generate dataset** with guardrails (`generate_dataset()`)
2. **Load real reviews** (`load_real_reviews()`)
3. **Generate quality report** (`generate_quality_report()`)
4. **Save datasets & metrics** (`save_dataset()`)

### 3.1. Outputs

- **Synthetic dataset**: `output/synthetic_reviews.csv`
  - `text`, `rating`, `persona`, `style`, `model`, `generation_time`, `timestamp`
- **Rejected samples**: `output/rejected_samples.csv`
  - Includes `rejection_reasons`
- **Metrics**: `output/metrics.json`
  - Total generated/rejected, per‑model stats, times
- **Quality report**: `output/quality_report.md`
  - Dataset overview, diversity metrics, bias metrics, synthetic vs real comparison, per‑model performance
- **Logs**: `output/logs/`
  - `created_reviews.log` – per‑review log for accepted samples
  - `failed_reviews.log` – rejected samples + errors with reasons

CLI only – no HTTP server; the script is the primary interface.

---

## 4. Configuration

### 4.1. `config/config.yaml`

- **Generation**
  - `target_samples`: number of reviews to generate (e.g. 400)
  - `models`:
    - `name`: logical name in metrics/logs
    - `provider`: `"groq"` and `"cerebras"`
    - `model`: provider‑specific model id (e.g. `llama-3.3-70b-versatile`, `gpt-oss-120b`)
    - `temperature`, `max_tokens`, `weight`
  - `rating_distribution`: probability per star rating (1–5)
  - `characteristics`:
    - `min_length`, `max_length`
    - `styles`: weights for `detailed`, `concise`, `emotional`, `technical`
    - `elements`: probabilities for including course aspects (content, instructor, etc.)

- **Quality**
  - `diversity`: min vocab diversity, max semantic similarity, min length variance
  - `bias`: max sentiment skew, keyword overuse thresholds
  - `realism`:
    - `require_domain_keywords`, `min_domain_keywords`
    - `check_rating_sentiment_match`, `sentiment_rating_tolerance`
  - `rejection`:
    - `max_regeneration_attempts`
    - `enable_auto_regeneration`

- **Comparison**
  - `real_reviews_path`: `data/real_reviews.jsonl`
  - `metrics`: list of comparison metrics to report

- **Output**
  - Paths for dataset, report, rejected samples, metrics
  - Logging configuration (log dir, log file names)

### 4.2. Personas & Prompts

- `config/personas.yaml` – personas (career switcher, student, professional upskiller, etc.) with:
  - age range, motivation, experience level, priorities, pain points, review style, typical concerns, weights
- `config/prompts.yaml` – controls the **style and diversity** of reviews:
  - `system_prompts.base` – global system instructions for authenticity
  - `generation_templates` – `detailed_review`, `concise_review`, `emotional_review`, `technical_review`
    - Updated to enforce **unique openings**, **concrete details**, **persona‑specific voice**
  - `techical_aspects`, `sentiment_guidelines`, `anti_patterns`

You can tweak these YAML files without touching Python code to change behaviour.

---

## 5. Design & Architecture

### 5.1. Multi‑Provider Model Abstraction

- `setup_llms()`:
  - Initializes Groq client (OpenAI‑compatible)
  - Initializes Cerebras client (OpenAI‑compatible chat completions)
  - Builds `self.llms` as a mapping: model name → `{client, client_type, config}`
- `select_model()` chooses a model per review based on weights.

**Trade‑off**: Use OpenAI‑compatible APIs for both providers → simpler code, but ties us to providers that support that interface.

### 5.2. Prompt & Persona Layer

- `build_prompt()`:
  - Chooses a prompt template based on style
  - Injects persona context, priorities, concerns, domain keywords, and technical aspects
  - Uses improved prompts to enforce uniqueness and concreteness

**Trade‑off**: Prompts are more verbose and strict, but this greatly reduces generic, low‑quality outputs (especially for Llama).

### 5.3. Quality Guardrails

Implemented in `check_quality()` and supporting methods:

- **Length checks** – reject too short/long reviews.
- **Anti‑pattern checks** – rejects obviously templated or over‑used phrases.
- **Domain realism** – requires a minimum number of EdTech domain keywords from `data/domain_keywords.txt`.
- **Sentiment–rating alignment** – uses TextBlob polarity vs expected sentiment per rating; tolerant threshold to avoid over‑rejecting.
- **Diversity metrics** – computed over accepted reviews:
  - Vocab diversity, semantic similarity (TF‑IDF + cosine), length statistics.
- **Bias detection** – rating distribution vs configured distribution, keyword overuse.

**Trade‑off**: Stronger guardrails can increase rejection and regeneration cost, but they improve dataset quality and realism.

### 5.4. Retry & Rate‑Limit Handling

- `generate_review()`:
  - Retries each model call up to `max_retries` with exponential backoff.
  - Validates response structure and text length before accepting.
- `generate_dataset()`:
  - Per‑review retry loop (quality‑aware)
  - Tracks rate‑limit‑like errors and backs off using pause logic.

**Trade‑off**: More robust to flaky API behaviour, at the cost of slightly higher latency per sample under failure conditions.

---

## 6. Limitations & Trade‑offs

- **Model choice**:
  - Groq (Llama‑3.3‑70B) and Cerebras (GPT‑OSS‑120B) were chosen for **cost‑effective, fast** open‑weights‑style models.
  - Some closed‑source models (e.g. GPT‑4 class) might produce even higher‑quality text but at higher cost.

- **Hardware / API constraints**:
  - All generation happens via cloud APIs (Groq + Cerebras), no local GPU required.
  - Throughput and latency are bounded by free‑tier / API limits (especially if you hit rate limits).

- **Sentiment detection**:
  - Uses TextBlob, which is simple and language‑agnostic but not as strong as modern transformer‑based sentiment models.
  - Tolerance is tuned to reduce false positives; extreme mismatches are still caught.

- **Real reviews scope**:
  - Real reviews are from one main source (Coursera ML, Marketing,DL courses); domain is EdTech but not all course types are covered.
  - Comparison is statistical, not human‑evaluated.

---

## 7. How This Meets the Assignment Requirements

- **Generation**:
  - 400 samples (configurable 300–500+)
  - YAML‑driven personas, rating distribution, review characteristics
  - At least 2 different models/providers (Groq + Cerebras)

- **Quality Guardrails**:
  - Diversity metrics: vocab diversity, semantic similarity, length stats
  - Bias detection: sentiment skew, rating distribution, keyword overuse
  - Domain realism: domain keywords, sentiment–rating checks
  - Automated rejection/regeneration for low‑quality or mismatched samples

- **Engineering**:
  - CLI interface (`python app.py`)
  - Quality report (`output/quality_report.md`)
  - Comparison vs 30–50 real reviews (`data/real_reviews.jsonl`)
  - Per‑model quality/time metrics in `metrics.json` + report

- **Deliverables**:
  - Codebase ready for GitHub repo
  - Generated dataset + quality scores (`synthetic_reviews.csv`, `rejected_samples.csv`, `metrics.json`)
  - Quality report: metrics, synthetic vs real, model performance
  - This `README.md` explains setup, design, trade‑offs, limitations.




