"""
Synthetic Data Generator for EdTech Course Reviews
Uses LangChain for LLM orchestration and quality guardrails
"""
import yaml
import json
import csv
import random
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
from collections import Counter
import re
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, SystemMessage

# For quality metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textblob import TextBlob
import os
import pandas as pd
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import httpx
try:
    from cerebras.cloud.sdk import Cerebras
    CEREBRAS_AVAILABLE = True
except ImportError:
    CEREBRAS_AVAILABLE = False


load_dotenv(".env")
random.seed(42)
np.random.seed(42)

# Get API keys and strip whitespace/quotes that might be in .env file
groq_api_key = os.getenv("GROQ_API_KEY", "").strip().strip('"').strip("'") if os.getenv("GROQ_API_KEY") else None
cerebras_api_key = os.getenv("CEREBRAS_API_KEY", "").strip().strip('"').strip("'") if os.getenv("CEREBRAS_API_KEY") else None

class SyntheticReviewGenerator:
    def __init__(self, config_path: str = "./config/config.yaml"):
        """Initialize the generator with configuration"""
        self.config = self.load_config(config_path)
        self.personas = self.load_yaml("./config/personas.yaml")
        self.prompts = self.load_yaml("./config/prompts.yaml")
        self.domain_keywords = self.load_keywords("./data/domain_keywords.txt")
        
        # Initialize LLMs
        self.llms = self.setup_llms()
        
        # Storage for generated reviews
        self.generated_reviews = []
        self.rejected_reviews = []
        self.real_reviews = []
        
        # Metrics tracking
        self.generation_metrics = {
            'total_generated': 0,
            'total_rejected': 0,
            'generation_times': [],
            'total_tokens': 0,          # aggregate tokens across all models (if usage available)
            'model_stats': {}           # per-model: count, times, tokens
        }
        
        # Setup logging
        self.setup_logging()
        
        print("✓ Synthetic Review Generator initialized")
        print(f"✓ Target samples: {self.config['generation']['target_samples']}")
        print(f"✓ Models: {[m['name'] for m in self.config['generation']['models']]}")
    
    def load_config(self, path: str) -> Dict:
        """Load YAML configuration"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_yaml(self, path: str) -> Dict:
        """Load YAML file"""
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def load_keywords(self, path: str) -> List[str]:
        """Load domain keywords from text file"""
        with open(path, 'r') as f:
            keywords = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        return keywords
    
    def setup_logging(self):
        """Setup logging for created and failed reviews"""
        # Create logs directory if it doesn't exist
        log_dir = Path(self.config['output'].get('log_dir', 'output/logs'))
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Get log file paths from config
        created_log_path = self.config['output'].get('created_reviews_log', str(log_dir / 'created_reviews.log'))
        failed_log_path = self.config['output'].get('failed_reviews_log', str(log_dir / 'failed_reviews.log'))
        
        # Setup logger for created reviews
        self.created_logger = logging.getLogger('created_reviews')
        self.created_logger.setLevel(logging.INFO)
        self.created_logger.handlers = []  # Clear existing handlers
        
        created_handler = logging.FileHandler(created_log_path, encoding='utf-8')
        created_handler.setLevel(logging.INFO)
        created_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        created_handler.setFormatter(created_formatter)
        self.created_logger.addHandler(created_handler)
        self.created_logger.propagate = False
        
        # Setup logger for failed reviews
        self.failed_logger = logging.getLogger('failed_reviews')
        self.failed_logger.setLevel(logging.INFO)
        self.failed_logger.handlers = []  # Clear existing handlers
        
        failed_handler = logging.FileHandler(failed_log_path, encoding='utf-8')
        failed_handler.setLevel(logging.INFO)
        failed_formatter = logging.Formatter(
            '%(asctime)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        failed_handler.setFormatter(failed_formatter)
        self.failed_logger.addHandler(failed_handler)
        self.failed_logger.propagate = False
        
        # Store log paths for reference
        self.created_log_path = created_log_path
        self.failed_log_path = failed_log_path
        
        print(f"✓ Logging initialized")
        print(f"  Created reviews log: {created_log_path}")
        print(f"  Failed reviews log: {failed_log_path}")
    
    def setup_llms(self) -> Dict:
        """Initialize LLM providers - Groq and Cerebras"""
        llms = {}
        
        # Access module-level API keys
        global groq_api_key, cerebras_api_key
        
        # Initialize Groq client
        groq_client = None
        if groq_api_key:
            try:
                import httpx
                http_client = httpx.Client(timeout=60.0, transport=httpx.HTTPTransport(retries=3))
                groq_client = OpenAI(
                    base_url="https://api.groq.com/openai/v1",
                    api_key=groq_api_key,
                    http_client=http_client,
                )
                print("✓ Groq client initialized")
            except Exception as e:
                try:
                    groq_client = OpenAI(
                        base_url="https://api.groq.com/openai/v1",
                        api_key=groq_api_key,
                    )
                    print("✓ Groq client initialized (simple)")
                except Exception as e2:
                    print(f"⚠ Groq initialization failed: {e2}")
                    groq_client = None
        
        # Initialize Cerebras client
        cerebras_client = None
        if cerebras_api_key and CEREBRAS_AVAILABLE:
            try:
                cerebras_client = Cerebras(api_key=cerebras_api_key)
                print("✓ Cerebras client initialized")
            except Exception as e:
                print(f"⚠ Cerebras initialization failed: {e}")
                cerebras_client = None
        elif not CEREBRAS_AVAILABLE:
            print("⚠ Cerebras SDK not available. Install with: pip install cerebras-cloud-sdk")
        
        # Map providers to clients
        provider_map = {
            'groq': groq_client,
            'meta-llama': groq_client,
            'cerebras': cerebras_client,
        }
        
        # Store client reference for each model
        for model_config in self.config['generation']['models']:
            provider = model_config.get('provider', 'groq').lower()
            
            # Get client from provider map
            client = provider_map.get(provider)
            
            if client is None:
                print(f"⚠ Skipping model '{model_config['name']}' - Provider '{provider}' not available")
                continue
            
            # Both Groq and Cerebras use OpenAI-compatible API
            client_type = 'openai'
            
            llms[model_config['name']] = {
                'client': client,
                'client_type': client_type,
                'config': model_config
            }
        
        if not llms:
            raise ValueError(
                "No models could be initialized! Please check:\n"
                "1. At least one API key is set (GROQ_API_KEY or CEREBRAS_API_KEY)\n"
                "2. Your config.yaml has models configured\n"
                "3. The provider names match your API keys\n"
                "4. For Cerebras: pip install cerebras-cloud-sdk"
            )
        
        print(f"\n✓ Initialized {len(llms)} model(s) successfully")
        return llms

        
    def select_model(self) -> Tuple[str, object]:
        """Select model based on weights"""
        models = self.config['generation']['models']
        weights = [m['weight'] for m in models]
        selected = random.choices(models, weights=weights, k=1)[0]
        return selected['name'], self.llms[selected['name']]
    
    def select_persona(self) -> Dict:
        """Select persona based on weights"""
        personas = self.personas['personas']
        weights = [p['weight'] for p in personas]
        return random.choices(personas, weights=weights, k=1)[0]
    
    def select_rating(self) -> int:
        """Select rating based on distribution"""
        ratings = list(self.config['generation']['rating_distribution'].keys())
        weights = list(self.config['generation']['rating_distribution'].values())
        return random.choices(ratings, weights=weights, k=1)[0]
    
    def select_review_style(self) -> str:
        """Select review style based on distribution"""
        styles = self.config['generation']['characteristics']['styles']
        style_names = list(styles.keys())
        weights = list(styles.values())
        return random.choices(style_names, weights=weights, k=1)[0]
    
    def build_prompt(self, persona: Dict, rating: int, style: str) -> str:
        """Build generation prompt from templates"""
        # Select appropriate template
        if style == 'detailed':
            template_config = self.prompts['generation_templates']['detailed_review']
        elif style == 'concise':
            template_config = self.prompts['generation_templates']['concise_review']
        elif style == 'emotional':
            template_config = self.prompts['generation_templates']['emotional_review']
        else:  # technical
            template_config = self.prompts['generation_templates']['technical_review']
        
        # Random course category
        course_category = random.choice(self.config['course_categories'])
        
        # Select review elements to include
        elements_config = self.config['generation']['characteristics']['elements']
        selected_elements = [elem for elem, prob in elements_config.items() 
                           if random.random() < prob]
        
        # Build the prompt
        prompt = template_config['template'].format(
            persona_name=persona['name'],
            persona_description=persona['description'],
            motivation=persona['characteristics']['motivation'],
            rating=rating,
            review_style=persona['characteristics']['review_style'],
            course_category=course_category,
            priorities=', '.join(persona['characteristics']['priorities']),
            typical_concerns=', '.join(persona['characteristics']['typical_concerns']),
            review_elements=', '.join(selected_elements),
            experience_level=persona['characteristics']['experience_level'],
            word_count_range=template_config.get('word_count_ranges', {}).get('medium', '100-200 words'),
            technical_aspects=', '.join(self.prompts['technical_aspects'][:3])
        )
        
        return prompt
    
    def generate_review(self) -> Dict:
        """Generate a single review with improved error handling"""
        start_time = time.time()
        
        # Select parameters
        persona = self.select_persona()
        rating = self.select_rating()
        style = self.select_review_style()
        model_name, llm = self.select_model()
        
        # Build prompt
        prompt = self.build_prompt(persona, rating, style)
        system_prompt = self.prompts['system_prompts']['base']
        
        # llm is now a dict with 'client' and 'config'
        client = llm['client']
        config = llm['config']
        provider = config.get('provider', 'groq').lower()
        
        # Increase max_tokens to reduce short responses
        max_tokens = max(config.get('max_tokens', 300), 500)
        
        # Add retry logic with exponential backoff
        max_retries = 3
        base_delay = 1.0
        
        total_tokens = None

        for attempt in range(max_retries):
            try:
                # Make API call
                if provider == 'cerebras':
                    response = client.chat.completions.create(
                        model=config['model'],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=config['temperature'],
                        max_completion_tokens=max_tokens,
                        top_p=1,
                        stream=False
                    )
                else:  # groq and others
                    response = client.chat.completions.create(
                        model=config['model'],
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=config['temperature'],
                        max_tokens=max_tokens
                    )
                
                # Validate response structure
                if not response:
                    raise ValueError("No response received from model")
                
                if not hasattr(response, 'choices') or not response.choices:
                    raise ValueError("Response has no choices")
                
                if not response.choices[0].message:
                    raise ValueError("Response choice has no message")
                
                if response.choices[0].message.content is None:
                    raise ValueError("Message content is None")
                
                # Extract token usage if available (OpenAI-compatible responses)
                usage = getattr(response, "usage", None)
                if usage is not None:
                    try:
                        total_tokens = getattr(usage, "total_tokens", None)
                        if total_tokens is None:
                            prompt_toks = getattr(usage, "prompt_tokens", 0)
                            completion_toks = getattr(usage, "completion_tokens", 0)
                            total_tokens = (prompt_toks or 0) + (completion_toks or 0)
                    except Exception:
                        total_tokens = None

                # Extract and validate review text
                review_text = response.choices[0].message.content.strip()
                
                if not review_text:
                    raise ValueError("Empty review text after stripping")
                
                # Check minimum length early
                word_count = len(review_text.split())
                min_len = self.config['generation']['characteristics']['min_length']
                
                if word_count < min_len:
                    raise ValueError(f"Generated text too short ({word_count} words, need {min_len})")
                
                # Success - create review object
                generation_time = time.time() - start_time
                
                review = {
                    'text': review_text,
                    'rating': rating,
                    'persona': persona['id'],
                    'style': style,
                    'model': model_name,
                    'generation_time': generation_time,
                    'timestamp': datetime.now().isoformat(),
                    'total_tokens': total_tokens
                }
                
                return review
                
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                is_rate_limit = 'rate_limit' in error_msg.lower() or '429' in error_msg
                
                # If rate limit or last attempt, raise the error
                if is_rate_limit or attempt == max_retries - 1:
                    raise
                
                # Otherwise, wait and retry with exponential backoff
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠ Attempt {attempt + 1} failed: {error_msg[:100]}... Retrying in {delay:.1f}s")
                time.sleep(delay)
        
        # Should not reach here, but just in case
        raise ValueError("Failed to generate review after all retries")

    
    def check_quality(self, review: Dict) -> Tuple[bool, List[str]]:
        """Check if review meets quality standards"""
        reasons = []
        text = review['text']
        rating = review['rating']
        
        # Check length
        word_count = len(text.split())
        min_len = self.config['generation']['characteristics']['min_length']
        max_len = self.config['generation']['characteristics']['max_length']
        
        if word_count < min_len:
            reasons.append(f"Too short ({word_count} words)")
        if word_count > max_len:
            reasons.append(f"Too long ({word_count} words)")
        
        # Check for anti-patterns
        text_lower = text.lower()
        for pattern in self.prompts['anti_patterns']:
            if pattern.lower() in text_lower:
                reasons.append(f"Contains anti-pattern: '{pattern}'")
        
        # Check domain keywords
        if self.config['quality']['realism']['require_domain_keywords']:
            keyword_count = sum(1 for kw in self.domain_keywords if kw.lower() in text_lower)
            min_keywords = self.config['quality']['realism']['min_domain_keywords']
            if keyword_count < min_keywords:
                reasons.append(f"Insufficient domain keywords ({keyword_count}/{min_keywords})")
        
        # Check sentiment-rating alignment
        if self.config['quality']['realism']['check_rating_sentiment_match']:
            sentiment = TextBlob(text).sentiment.polarity
            expected_sentiment = self.get_expected_sentiment(rating)
            # Loosen tolerance to reduce false rejects for slightly off sentiment
            tolerance = max(self.config['quality']['realism']['sentiment_rating_tolerance'], 0.6)
            
            if abs(sentiment - expected_sentiment) > tolerance:
                reasons.append(f"Sentiment mismatch (sentiment: {sentiment:.2f}, expected: {expected_sentiment:.2f})")
        
        return len(reasons) == 0, reasons
    
    def get_expected_sentiment(self, rating: int) -> float:
        """Get expected sentiment polarity for rating"""
        sentiment_map = {1: -0.5, 2: -0.2, 3: 0.0, 4: 0.3, 5: 0.5}
        return sentiment_map.get(rating, 0.0)
    
    def calculate_diversity_metrics(self, reviews: List[Dict]) -> Dict:
        """Calculate diversity metrics across reviews"""
        if not reviews:
            return {}
        
        texts = [r['text'] for r in reviews]
        
        # Vocabulary diversity
        all_words = ' '.join(texts).lower().split()
        unique_words = set(all_words)
        vocab_diversity = len(unique_words) / len(all_words) if all_words else 0
        
        # Semantic similarity
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(texts)
        similarities = cosine_similarity(tfidf_matrix)
        
        # Get upper triangle (exclude diagonal)
        upper_tri = similarities[np.triu_indices_from(similarities, k=1)]
        avg_similarity = float(np.mean(upper_tri)) if len(upper_tri) > 0 else 0
        max_similarity = float(np.max(upper_tri)) if len(upper_tri) > 0 else 0
        
        # Length variance
        lengths = [len(r['text'].split()) for r in reviews]
        length_std = np.std(lengths)
        length_mean = np.mean(lengths)
        length_cv = length_std / length_mean if length_mean > 0 else 0
        
        return {
            'vocabulary_diversity': vocab_diversity,
            'avg_semantic_similarity': avg_similarity,
            'max_semantic_similarity': max_similarity,
            'length_coefficient_of_variation': length_cv,
            'avg_length': length_mean,
            'std_length': length_std
        }
    
    def detect_bias(self, reviews: List[Dict]) -> Dict:
        """Detect bias in generated reviews"""
        if not reviews:
            return {}
        
        # Rating distribution
        ratings = [r['rating'] for r in reviews]
        rating_counts = Counter(ratings)
        actual_dist = {r: rating_counts[r] / len(ratings) for r in range(1, 6)}
        expected_dist = self.config['generation']['rating_distribution']
        
        # Calculate skew
        sentiment_skew = sum(abs(actual_dist.get(r, 0) - expected_dist.get(r, 0)) 
                           for r in range(1, 6))
        
        # Keyword overuse
        texts = [r['text'].lower() for r in reviews]
        keyword_frequencies = {}
        for keyword in self.domain_keywords[:50]:  # Check top 50 keywords
            count = sum(1 for text in texts if keyword.lower() in text)
            keyword_frequencies[keyword] = count / len(texts)
        
        overused_keywords = {k: v for k, v in keyword_frequencies.items() 
                           if v > self.config['quality']['bias']['max_keyword_frequency']}
        
        return {
            'sentiment_skew': sentiment_skew,
            'rating_distribution': actual_dist,
            'overused_keywords': overused_keywords,
            'bias_detected': sentiment_skew > self.config['quality']['bias']['max_sentiment_skew']
        }
    

    def generate_dataset(self):
        """Generate the full dataset with quality checks and rate limit handling"""
        target = self.config['generation']['target_samples']
        print(f"\n{'='*80}")
        print(f"Starting generation of {target} reviews...")
        print(f"{'='*80}\n")
        
        max_attempts = self.config['quality']['rejection'].get('max_regeneration_attempts', 3)
        auto_regen = self.config['quality']['rejection'].get('enable_auto_regeneration', True)
        
        # Rate limit tracking
        rate_limit_cooldown = 0
        consecutive_failures = 0
        max_consecutive_failures = 10

        while len(self.generated_reviews) < target:
            # Check if we hit too many consecutive failures
            if consecutive_failures >= max_consecutive_failures:
                print(f"\n⚠ Hit {max_consecutive_failures} consecutive failures. Pausing for 60 seconds...")
                time.sleep(60)
                consecutive_failures = 0
            
            attempts = 0
            success = False

            while attempts < max_attempts and not success:
                try:
                    # Apply rate limit cooldown if needed
                    if rate_limit_cooldown > 0:
                        time.sleep(rate_limit_cooldown)
                        rate_limit_cooldown = 0
                    
                    # Generate review
                    review = self.generate_review()
                    self.generation_metrics['total_generated'] += 1
                    self.generation_metrics['generation_times'].append(review['generation_time'])
                    
                    # Track tokens if available
                    tokens = review.get('total_tokens')
                    if tokens:
                        self.generation_metrics['total_tokens'] = self.generation_metrics.get('total_tokens', 0) + tokens
                    
                    # Track model stats
                    model = review['model']
                    if model not in self.generation_metrics['model_stats']:
                        self.generation_metrics['model_stats'][model] = {
                            'count': 0,
                            'times': [],
                            'tokens': 0
                        }
                    self.generation_metrics['model_stats'][model]['count'] += 1
                    self.generation_metrics['model_stats'][model]['times'].append(review['generation_time'])
                    if tokens:
                        self.generation_metrics['model_stats'][model]['tokens'] += tokens
                    
                    # Check quality
                    passed, reasons = self.check_quality(review)
                    
                    if passed:
                        self.generated_reviews.append(review)
                        consecutive_failures = 0  # Reset on success
                        
                        # Log successful review creation
                        log_message = (
                            f"Review #{len(self.generated_reviews)} | "
                            f"Model: {review['model']} | "
                            f"Rating: {review['rating']}⭐ | "
                            f"Persona: {review['persona']} | "
                            f"Style: {review['style']} | "
                            f"Time: {review['generation_time']:.2f}s | "
                            f"Length: {len(review['text'].split())} words\n"
                            f"Text: {review['text'][:200]}..." if len(review['text']) > 200 else f"Text: {review['text']}"
                        )
                        self.created_logger.info(log_message)
                        
                        print(f"✓ Generated {len(self.generated_reviews)}/{target} | "
                            f"Model: {review['model']} | "
                            f"Rating: {review['rating']}⭐ | "
                            f"Time: {review['generation_time']:.2f}s")
                        success = True
                    else:
                        self.rejected_reviews.append({**review, 'rejection_reasons': reasons})
                        self.generation_metrics['total_rejected'] += 1
                        
                        # Log failed review
                        log_message = (
                            f"Rejected Review | "
                            f"Model: {review['model']} | "
                            f"Rating: {review['rating']}⭐ | "
                            f"Persona: {review['persona']} | "
                            f"Time: {review['generation_time']:.2f}s | "
                            f"Reasons: {', '.join(reasons)}\n"
                            f"Text: {review['text'][:200]}..." if len(review['text']) > 200 else f"Text: {review['text']}"
                        )
                        self.failed_logger.info(log_message)
                        
                        print(f"✗ Rejected | Reasons: {', '.join(reasons[:2])}")
                        
                        if auto_regen:
                            attempts += 1
                            continue
                        else:
                            break
                
                except Exception as e:
                    error_msg = str(e)
                    attempts += 1
                    consecutive_failures += 1
                    
                    # Handle rate limit errors specially
                    if 'rate_limit' in error_msg.lower() or '429' in error_msg:
                        # Extract wait time from error message if available
                        import re
                        time_match = re.search(r'(\d+)m(\d+(?:\.\d+)?)?s', error_msg)
                        if time_match:
                            minutes = int(time_match.group(1))
                            seconds = float(time_match.group(2) or 0)
                            wait_time = minutes * 60 + seconds + 5  # Add 5 second buffer
                            print(f"⚠ Rate limit hit. Waiting {wait_time:.0f} seconds...")
                            time.sleep(wait_time)
                            rate_limit_cooldown = 0
                            attempts -= 1  # Don't count rate limits as attempts
                            consecutive_failures -= 1
                            continue
                        else:
                            # Generic rate limit cooldown
                            rate_limit_cooldown = 60
                            print(f"⚠ Rate limit hit. Will wait 60s before next request.")
                            attempts -= 1  # Don't count rate limits as attempts
                            consecutive_failures -= 1
                            continue
                    
                    print(f"✗ Error (attempt {attempts}/{max_attempts}): {error_msg[:100]}")
                    
                    # Log the error
                    if hasattr(self, "failed_logger"):
                        self.failed_logger.error(
                            f"Generation Error | Attempt {attempts}/{max_attempts} | "
                            f"{error_msg} | Timestamp: {datetime.now().isoformat()}"
                        )
                    
                    if auto_regen and attempts < max_attempts:
                        # Small delay between retries
                        time.sleep(2)
                        continue
                    else:
                        break

            # If attempts exhausted without success, continue to next
            if not success:
                continue
        
        print(f"\n{'='*80}")
        print(f"✓ Generation complete!")
        print(f"  Generated: {len(self.generated_reviews)}")
        print(f"  Rejected: {len(self.rejected_reviews)}")
        print(f"  Acceptance rate: {len(self.generated_reviews) / self.generation_metrics['total_generated'] * 100:.1f}%")
        print(f"{'='*80}\n")
    
    def load_real_reviews(self):
        """Load real reviews for comparison"""
        path = self.config['comparison']['real_reviews_path']
        if not Path(path).exists():
            print(f"⚠ Real reviews file not found: {path}")
            return
        
        # Load JSON file and extract reviews list
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Handle both formats: direct list or nested in 'reviews' key
        if isinstance(data, list):
            self.real_reviews = data
        elif isinstance(data, dict) and 'reviews' in data:
            self.real_reviews = data['reviews']
        else:
            # Try pandas as fallback
            df = pd.read_json(path)
            if 'reviews' in df.columns:
                self.real_reviews = df['reviews'].iloc[0] if len(df) > 0 else []
            else:
                self.real_reviews = df.to_dict(orient='records')
        
        print(f"✓ Loaded {len(self.real_reviews)} real reviews")
    
    def compare_with_real(self) -> Dict:
        """Compare synthetic reviews with real ones"""
        if not self.real_reviews:
            return {}
        
        print("\nComparing with real reviews...")
        
        synthetic_texts = [r['text'] for r in self.generated_reviews]
        # Handle different possible field names for review text
        real_texts = []
        for r in self.real_reviews:
            if isinstance(r, dict):
                text = r.get('review_text') or r.get('text') or r.get('content', '')
                if text:
                    real_texts.append(text)
        
        if not real_texts:
            print("⚠ No valid review texts found in real reviews data")
            return {}
        
        # Vocabulary overlap
        synthetic_vocab = set(' '.join(synthetic_texts).lower().split())
        real_vocab = set(' '.join(real_texts).lower().split())
        vocab_overlap = len(synthetic_vocab & real_vocab) / len(synthetic_vocab | real_vocab)
        
        # Length distribution comparison
        synthetic_lengths = [len(t.split()) for t in synthetic_texts]
        real_lengths = [len(t.split()) for t in real_texts]
        
        # Sentiment distribution
        synthetic_sentiments = [TextBlob(t).sentiment.polarity for t in synthetic_texts]
        real_sentiments = [TextBlob(t).sentiment.polarity for t in real_texts]
        
        # Rating distribution
        synthetic_ratings = [r['rating'] for r in self.generated_reviews]
        real_ratings = [int(float(r['rating'])) for r in self.real_reviews if 'rating' in r]
        
        comparison = {
            'vocabulary_overlap': vocab_overlap,
            'length_comparison': {
                'synthetic_mean': np.mean(synthetic_lengths),
                'real_mean': np.mean(real_lengths),
                'synthetic_std': np.std(synthetic_lengths),
                'real_std': np.std(real_lengths)
            },
            'sentiment_comparison': {
                'synthetic_mean': np.mean(synthetic_sentiments),
                'real_mean': np.mean(real_sentiments),
                'synthetic_std': np.std(synthetic_sentiments),
                'real_std': np.std(real_sentiments)
            },
            'rating_comparison': {
                'synthetic_dist': dict(Counter(synthetic_ratings)),
                'real_dist': dict(Counter(real_ratings))
            }
        }
        
        return comparison
    
    def generate_quality_report(self):
        """Generate comprehensive quality report"""
        print("\nGenerating quality report...")
        
        # Calculate metrics
        diversity = self.calculate_diversity_metrics(self.generated_reviews)
        bias = self.detect_bias(self.generated_reviews)
        comparison = self.compare_with_real()
        
        # Create report
        report_lines = [
            "# Synthetic Review Dataset - Quality Report",
            f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"\n## Dataset Overview",
            f"- Total synthetic reviews: {len(self.generated_reviews)}",
            f"- Total rejected: {len(self.rejected_reviews)}",
            f"- Acceptance rate: {len(self.generated_reviews) / self.generation_metrics['total_generated'] * 100:.1f}%",
            f"- Real reviews for comparison: {len(self.real_reviews)}",
            
            f"\n## Generation Performance",
            f"- Average generation time: {np.mean(self.generation_metrics['generation_times']):.2f}s",
            f"- Total generation time: {sum(self.generation_metrics['generation_times']) / 60:.1f} minutes",
            f"- Total tokens (approx): {self.generation_metrics.get('total_tokens', 0)}",
            
            f"\n### Model Performance",
        ]
        
        for model, stats in self.generation_metrics['model_stats'].items():
            avg_time = np.mean(stats['times'])
            model_tokens = stats.get('tokens', 0)
            avg_tokens = model_tokens / stats['count'] if stats['count'] > 0 else 0
            report_lines.append(
                f"- {model}: {stats['count']} reviews, "
                f"avg time {avg_time:.2f}s, "
                f"total tokens {model_tokens}, avg tokens/review {avg_tokens:.0f}"
            )
        
        report_lines.extend([
            f"\n## Diversity Metrics",
            f"- Vocabulary diversity: {diversity.get('vocabulary_diversity', 0):.3f}",
            f"- Average semantic similarity: {diversity.get('avg_semantic_similarity', 0):.3f}",
            f"- Max semantic similarity: {diversity.get('max_semantic_similarity', 0):.3f}",
            f"- Length coefficient of variation: {diversity.get('length_coefficient_of_variation', 0):.3f}",
            f"- Average review length: {diversity.get('avg_length', 0):.1f} words",
            
            f"\n## Bias Detection",
            f"- Sentiment skew: {bias.get('sentiment_skew', 0):.3f}",
            f"- Bias detected: {'Yes' if bias.get('bias_detected') else 'No'}",
            f"- Overused keywords: {len(bias.get('overused_keywords', {}))}",
        ])
        
        if comparison:
            report_lines.extend([
                f"\n## Comparison with Real Reviews",
                f"- Vocabulary overlap: {comparison['vocabulary_overlap']:.3f}",
                f"\n### Length Distribution",
                f"- Synthetic mean: {comparison['length_comparison']['synthetic_mean']:.1f} words",
                f"- Real mean: {comparison['length_comparison']['real_mean']:.1f} words",
                f"\n### Sentiment Distribution",
                f"- Synthetic mean: {comparison['sentiment_comparison']['synthetic_mean']:.3f}",
                f"- Real mean: {comparison['sentiment_comparison']['real_mean']:.3f}",
            ])
        
        report_text = '\n'.join(report_lines)
        
        # Save report
        output_path = self.config['output']['quality_report_path']
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(f"✓ Quality report saved to: {output_path}")
        
        return report_text
    
    def save_dataset(self):
        """Save generated dataset to CSV"""
        output_path = self.config['output']['dataset_path']
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            if self.generated_reviews:
                writer = csv.DictWriter(f, fieldnames=self.generated_reviews[0].keys())
                writer.writeheader()
                writer.writerows(self.generated_reviews)
        
        print(f"✓ Dataset saved to: {output_path}")
        
        # Save rejected samples
        if self.rejected_reviews:
            rejected_path = self.config['output']['rejected_samples_path']
            with open(rejected_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=self.rejected_reviews[0].keys())
                writer.writeheader()
                writer.writerows(self.rejected_reviews)
            print(f"✓ Rejected samples saved to: {rejected_path}")
        
        # Save metrics
        metrics_path = self.config['output']['metrics_path']
        with open(metrics_path, 'w') as f:
            json.dump(self.generation_metrics, f, indent=2)
        print(f"✓ Metrics saved to: {metrics_path}")
    
    def run(self):
        """Run the complete pipeline"""
        print("\n" + "="*80)
        print("SYNTHETIC REVIEW GENERATOR - EdTech Domain")
        print("="*80)
        
        # Generate dataset
        self.generate_dataset()
        
        # Load real reviews
        self.load_real_reviews()
        
        # Generate report
        self.generate_quality_report()
        
        # Save everything
        self.save_dataset()
        
        print("\n" + "="*80)
        print("✓ Pipeline complete!")
        print("="*80 + "\n")


def main():
    """Main entry point"""
    generator = SyntheticReviewGenerator()
    generator.run()


if __name__ == "__main__":
    main()