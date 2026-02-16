Synthetic Persona Script: Installation & Usage Guide
This script automates the synthetic persona generation process covered in the main article. Instead of manually building persona cards from your data sources, the script does the heavy lifting: it synthesizes your Google Search Console data, survey responses, and customer conversation transcripts into complete persona cards with trackable prompts.
What The Script Generates
You'll get 2-3 persona cards (depending on data volume) that include:
●	Persona name and confidence score: Based on the amount of supporting data
●	Complete 5-field persona card: Job-to-be-done, constraints, success metric, decision criteria, and vocabulary
●	Trackable prompts: 15-30 prompts per persona across low, medium, and high intent levels
●	Evidence tracing: Documentation of which data sources support each persona field
●	Multiple output formats: Markdown (for LLMs), JSON (for programmatic use), HTML (for browser viewing)
The script solves the cold-start problem by turning your existing behavioral data into actionable personas without waiting weeks for traditional research.
Setup Instructions
Step 1: Access Google Colab
●	Navigate to colab.research.google.com
●	Create a new notebook or open an existing one
●	Click "Code" to create a new cell
●	Copy-paste the entire script into the cell (approximately 2,000 lines)
Step 2: Run The Script
Two options:
●	Scroll to the bottom and click the play button on the left side of the cell
●	Or use "Run All" from the Runtime menu
The script loads in under a minute and will prompt you for data files.
Optional: If you want the script to look a little nicer and come with a Wordcloud, add a code cell before the main script and copy/paste this:
!pip install tqdm scikit-learn matplotlib wordcloud
Step 3: Upload Your Data
The script accepts three types of input:
Required:
●	Google Search Console CSV export: Use the last 28 days, filtered by your target country. Export query data using this regex filter to capture question-type queries: (?i)^(who|what|why|how|when|where|which|can|does|is|are|should|guide|tutorial|course|learn|examples?|definition|meaning|checklist|framework|template|tips?|ideas?|best|top|lists?|comparison|vs|difference|benefits|advantages|alternatives)\b.*
Optional (but recommended for richer personas):
●	Survey data CSV: Newsletter surveys, customer feedback forms, onboarding questionnaires
●	Transcript files: Customer conversations, support calls, sales calls, user interviews (TXT format)
The more data sources you provide, the higher the confidence scores for each persona field.
Step 4: Filter Your Voice (Optional)
If you're uploading transcripts that include your own voice (customer calls, interviews), the script will prompt you to enter your name. This filters out your input so the personas reflect pure customer language patterns, not your facilitation questions.
You can enter multiple names if several team members appear in transcripts, or skip this step if you're only uploading one-sided data like support tickets.
Step 5: Process & Download
The script runs through approximately 10 processing steps:
1.	Data ingestion and cleaning
2.	Pattern extraction across sources
3.	Vocabulary analysis
4.	Job-to-be-done identification
5.	Constraint mapping
6.	Success metric synthesis
7.	Decision criteria extraction
8.	Persona clustering
9.	Prompt generation
10.	Evidence linking
After processing completes (typically 2-5 minutes), you'll see a summary and download options.
Available downloads:
●	Persona card Markdown files (for uploading to ChatGPT/Claude/Gemini)
●	Persona card JSON files (for programmatic use)
●	Persona card HTML files (for viewing in browser)
●	Evidence file (traces each persona field back to source data)
●	Run summary (documents what the script processed)
Download options:
●	Type specific numbers separated by commas (e.g., "1, 3, 6")
●	Type "all" to download everything
●	Type "skip" to just view the summary in Colab
For most use cases, download the Markdown file for your primary persona, plus the HTML files for all personas (for reference), plus the evidence file (for validation).
Step 6: Generate Trackable Prompts
Once you have your persona cards:
1.	Open the HTML file in your browser to review the complete persona profile
2.	Copy the "How to Generate Trackable Prompts" section from the persona card
3.	Open your LLM of choice (ChatGPT, Claude, or Gemini)
4.	Upload the Markdown file for that persona
5.	Paste the prompt generation instructions
6.	The LLM will output 15-30 prompts organized by intent level:
○	Low intent: Exploration and problem-space learning
○	Medium intent: Evaluation and solution comparison
○	High intent: Final decision and proof point validation
These prompts reflect how that specific persona segment would actually search, using their natural vocabulary and shaped by their real constraints.
Using Your Personas For Prompt Tracking
Feed these prompts into your AI search tracking tool of choice, organized by:
●	Persona (Enterprise IT Buyer vs. Individual User)
●	Intent level (Low vs. Medium vs. High)
●	Topic/product category
Track citation presence, ranking position, and coverage gaps over time. Re-run the same prompts weekly or bi-weekly to measure how content changes affect visibility across different user segments.
Maintenance & Updates
Synthetic personas stay accurate as long as your input data stays current. Set regeneration triggers:
●	Quarterly refresh using new GSC data and recent support tickets
●	Immediate refresh when major product changes ship
●	Immediate refresh when competitor dynamics shift (new entrant, positioning change)
●	Immediate refresh when you notice vocabulary shifts in customer conversations
Simply re-run the script with updated data files. The persona confidence scores will tell you if you have enough new data to warrant regeneration.
Troubleshooting
If you encounter issues:
●	Reply to your premium subscriber email
●	Comment on the article
●	Most issues stem from CSV formatting (ensure UTF-8 encoding) or file size limits in Colab
Expected total time: 5-10 minutes for initial setup and generation, 2-3 minutes for subsequent regenerations.
The script
"""
Synthetic Persona Generator v2.1
=================================
Generate data-driven subscriber personas from GSC queries, surveys, and call transcripts.

NEW in v2.1:
- Manual file selection (no auto-download)
- Enhanced specification requirements with provenance tracking
- Smarter persona clustering (only create multiple if truly differentiated)
- Specific, actionable content strategy recommendations
- Consistent persona card structure
- Improved vocabulary analysis
- Prompts displayed in output
"""

import io, os, re, sys, json, zipfile
from datetime import datetime
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional, Any
import pandas as pd
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    print(" Install tqdm for progress bars: !pip install tqdm")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import LatentDirichletAllocation
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print(" Install scikit-learn for clustering: !pip install scikit-learn")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print(" Install matplotlib for visualizations: !pip install matplotlib")

try:
    from wordcloud import WordCloud
    HAS_WORDCLOUD = True
except ImportError:
    HAS_WORDCLOUD = False
    print(" Install wordcloud for vocabulary clouds: !pip install wordcloud")

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
    print(" Not running in Google Colab. File upload and download features disabled.")

CONFIG = {
    "generate_visualizations": True,
    "generate_html": True,
    "examples": {"total": 5, "per_source": 2, "max_chars": 220},
    "vocab": {"min_df": 2, "top_n": 12},
    "min_evidence_rows": 50,
    "recommended_evidence_rows": 200,
    "speaker_filters": [],
    "max_personas": 3,
    "min_cluster_size": 50,
    "min_silhouette_score": 0.15,  # Minimum differentiation between clusters
}

# Context explanations for persona insights
INSIGHT_CONTEXT = {
    "jobs": {
        "measure": {
            "what": "Track and quantify performance metrics",
            "why": "Need to demonstrate ROI and understand what's working",
            "examples": ["Setting up analytics dashboards", "Tracking keyword rankings", "Measuring conversion rates"]
        },
        "optimize": {
            "what": "Improve existing processes or content performance",
            "why": "Want to get better results from current efforts",
            "examples": ["A/B testing content", "Improving page load speed", "Refining keyword targeting"]
        },
        "understand": {
            "what": "Learn new concepts or decode complex systems",
            "why": "Need foundational knowledge before taking action",
            "examples": ["Understanding algorithm updates", "Learning new frameworks", "Grasping industry trends"]
        },
        "implement": {
            "what": "Execute new strategies or tactics",
            "why": "Ready to act and need step-by-step guidance",
            "examples": ["Rolling out technical SEO fixes", "Launching new content formats", "Deploying automation"]
        },
        "report": {
            "what": "Communicate results to stakeholders",
            "why": "Need to justify efforts and secure continued buy-in",
            "examples": ["Creating executive dashboards", "Presenting quarterly results", "Documenting wins"]
        },
        "stay_current": {
            "what": "Keep up with industry changes and trends",
            "why": "Fear of falling behind or missing opportunities",
            "examples": ["Following algorithm updates", "Monitoring competitor moves", "Tracking AI developments"]
        }
    },
    "frustrations": {
        "measurement_gaps": {
            "what": "Inability to track or quantify important outcomes",
            "why": "Creates uncertainty about what's working and limits optimization",
            "impact": "Leads to gut-feel decisions rather than data-driven strategy",
            "examples": ["Can't attribute revenue to content", "No visibility into search intent shifts", "Unclear brand lift metrics"]
        },
        "tool_limitations": {
            "what": "Software can't do what you need it to",
            "why": "Forces manual workarounds or leaves gaps in workflow",
            "impact": "Wastes time and creates frustration with tech stack",
            "examples": ["Tool doesn't integrate with CMS", "Missing key data exports", "Requires coding to customize"]
        },
        "lack_guidance": {
            "what": "Unclear on best practices or how to approach a problem",
            "why": "No clear playbook or precedent to follow",
            "impact": "Analysis paralysis or wasted effort on wrong approaches",
            "examples": ["Don't know how to prioritize opportunities", "Unclear which metrics matter", "No framework for decisions"]
        },
        "stakeholder_pressure": {
            "what": "Leadership demands results or justification",
            "why": "Need to prove value and secure resources",
            "impact": "Time spent on reporting instead of execution",
            "examples": ["Proving SEO ROI to C-suite", "Justifying content budget", "Defending strategy choices"]
        }
    },
    "motivations": {
        "fomo": {
            "what": "Fear of missing out on opportunities or falling behind",
            "why": "Industry moves fast and competitors are always innovating",
            "drives": "Seeking early insights and competitive intelligence"
        },
        "efficiency": {
            "what": "Desire to accomplish more with less effort",
            "why": "Limited resources and increasing workload",
            "drives": "Looking for automation, shortcuts, and productivity hacks"
        },
        "impact": {
            "what": "Drive to create measurable business outcomes",
            "why": "Want to demonstrate value and advance career",
            "drives": "Seeking high-leverage tactics and result-focused strategies"
        },
        "mastery": {
            "what": "Aspiration to become an expert in the field",
            "why": "Professional growth and career advancement",
            "drives": "Looking for deep-dives, frameworks, and thought leadership"
        }
    },
    "decision_criteria": {
        "evidence": {
            "what": "Need proof that something works before trying it",
            "signals": "Values data, case studies, and research",
            "content_preferences": ["Case studies with data", "Research-backed articles", "Benchmark reports"]
        },
        "implementation": {
            "what": "Want clear steps and how-to guidance",
            "signals": "Action-oriented, ready to execute",
            "content_preferences": ["Step-by-step tutorials", "Implementation checklists", "Video walkthroughs"]
        },
        "templates": {
            "what": "Prefer ready-to-use frameworks and tools",
            "signals": "Time-pressed, wants to adapt existing solutions",
            "content_preferences": ["Spreadsheet templates", "Swipe files", "Process docs"]
        },
        "depth": {
            "what": "Seeks comprehensive, nuanced understanding",
            "signals": "Not satisfied with surface-level content",
            "content_preferences": ["Long-form deep-dives", "Technical analysis", "Primary research"]
        }
    }
}

def mask_pii(text: str) -> str:
    return re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', text)

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).strip()
    text = re.sub(r'\s+', ' ', text)
    text = mask_pii(text)
    return text

def truncate(text: str, max_chars: int = 200) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars-3] + "..."

def progress_wrapper(iterable, desc: str = "", total: int = None):
    if HAS_TQDM and (total is None or total > 5):
        return tqdm(iterable, desc=desc, total=total)
    return iterable

GSC_SYNONYMS = {
    "query": ["query", "queries", "top queries", "search query", "keyword"],
    "clicks": ["clicks"],
    "impressions": ["impressions"],
    "ctr": ["ctr", "click-through rate"],
    "position": ["position", "avg position", "average position"],
    "page": ["page", "url", "landing page"],
    "country": ["country"],
    "device": ["device"],
    "date": ["date"],
}

SURVEY_SYNONYMS = {
    "role": ["role", "title", "job title", "position", "what is your role"],
    "company_size": ["company size", "team size", "organization size"],
    "industry": ["industry", "sector", "vertical"],
    "resources": ["resources", "preferred resources", "favorite resources"],
    "formats": ["formats", "content formats", "preferred formats"],
    "challenges": ["challenges", "pain points", "frustrations", "problems"],
    "goals": ["goals", "objectives", "priorities"],
    "open_text": ["comments", "feedback", "additional thoughts"],
}

def detect_header_mapping(df: pd.DataFrame, synonyms: Dict[str, List[str]]) -> Dict[str, str]:
    mapping = {}
    cols_lower = {c: c.lower() for c in df.columns}
    for canonical, candidates in synonyms.items():
        for col_name, col_lower in cols_lower.items():
            if any(cand in col_lower for cand in candidates):
                mapping[col_name] = canonical
                break
    return mapping

def upload_file(prompt: str, required: bool = False) -> Optional[Dict[str, Any]]:
    if not IN_COLAB:
        print(f" {prompt} (skipped - not in Colab)")
        return None
    print(f"\n{prompt}")
    uploaded = files.upload()
    if not uploaded:
        if required:
            print(" No file uploaded (required)")
            return None
        else:
            print("⏭ Skipped")
            return None
    filename = list(uploaded.keys())[0]
    content = uploaded[filename]
    return {"filename": filename, "content": content, "size": len(content)}

def upload_multiple_files(prompt: str) -> List[Dict[str, Any]]:
    if not IN_COLAB:
        print(f" {prompt} (skipped - not in Colab)")
        return []
    print(f"\n{prompt}")
    print(" Tip: You can select multiple files at once")
    uploaded = files.upload()
    if not uploaded:
        print("⏭ No files uploaded")
        return []
    results = []
    for filename, content in uploaded.items():
        results.append({"filename": filename, "content": content, "size": len(content)})
    print(f" Uploaded {len(results)} file(s)")
    return results

def parse_transcript(text: str, speaker_filters: List[str] = None) -> List[Dict[str, str]]:
    utterances = []
    current_speaker = None
    current_text = []
    lines = text.split('\n')
    timestamp_pattern = re.compile(r'^\d{1,2}:\d{2}')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if timestamp_pattern.match(line):
            if current_speaker and current_text:
                full_text = ' '.join(current_text)
                if speaker_filters is None or any(sf.lower() in current_speaker.lower() for sf in speaker_filters):
                    utterances.append({"speaker": current_speaker, "text": clean_text(full_text)})
            parts = line.split(' - ', 1)
            if len(parts) == 2:
                current_speaker = parts[1].strip()
                current_text = []
        else:
            if current_speaker:
                current_text.append(line)
    if current_speaker and current_text:
        full_text = ' '.join(current_text)
        if speaker_filters is None or any(sf.lower() in current_speaker.lower() for sf in speaker_filters):
            utterances.append({"speaker": current_speaker, "text": clean_text(full_text)})
    return utterances

def extract_questions_from_transcripts(utterances: List[Dict[str, str]]) -> List[str]:
    questions = []
    for utt in utterances:
        text = utt["text"]
        sentences = re.split(r'[.!?]+', text)
        for sent in sentences:
            sent = sent.strip()
            if '?' in sent or any(sent.lower().startswith(q) for q in ["how", "what", "why", "when", "where", "who", "can", "should", "is", "do", "does"]):
                if len(sent) > 10:
                    questions.append(sent)
    return questions

def load_gsc_csv(file_content: bytes) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(io.BytesIO(file_content))
    mapping = detect_header_mapping(df, GSC_SYNONYMS)
    if not any(v == "query" for v in mapping.values()):
        raise ValueError("GSC CSV missing required column: 'query'. Please export with queries included.")
    df_renamed = df.rename(columns=mapping)
    print(f"   Read {len(df_renamed)} rows, {len(df_renamed.columns)} columns")
    print(f"    Mapped columns: {mapping}")
    return df_renamed, mapping

def load_survey_csv(file_content: bytes) -> Tuple[pd.DataFrame, Dict]:
    df = pd.read_csv(io.BytesIO(file_content))
    mapping = detect_header_mapping(df, SURVEY_SYNONYMS)
    df_renamed = df.rename(columns=mapping)
    print(f"   Read {len(df_renamed)} rows, {len(df_renamed.columns)} columns")
    print(f"    Mapped columns: {mapping}")
    return df_renamed, mapping

def load_transcripts(file_infos: List[Dict], speaker_filters: List[str] = None) -> Tuple[List[Dict], List[str]]:
    all_utterances = []
    all_questions = []
    for file_info in progress_wrapper(file_infos, desc="Parsing transcripts"):
        content = file_info["content"]
        if file_info["filename"].endswith('.zip'):
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                for name in zf.namelist():
                    if name.endswith('.txt'):
                        with zf.open(name) as f:
                            text = f.read().decode('utf-8', errors='ignore')
                            utterances = parse_transcript(text, speaker_filters)
                            all_utterances.extend(utterances)
                            all_questions.extend(extract_questions_from_transcripts(utterances))
        else:
            text = content.decode('utf-8', errors='ignore')
            utterances = parse_transcript(text, speaker_filters)
            all_utterances.extend(utterances)
            all_questions.extend(extract_questions_from_transcripts(utterances))
    print(f" Parsed {len(file_infos)} transcript file(s) → {len(all_utterances)} utterances, {len(all_questions)} questions")
    return all_utterances, all_questions

def build_evidence_corpus(gsc_df: pd.DataFrame, survey_df: pd.DataFrame = None, transcript_utterances: List[Dict] = None) -> pd.DataFrame:
    evidence_rows = []
    if "query" in gsc_df.columns:
        for idx, row in gsc_df.iterrows():
            query = clean_text(row["query"])
            if len(query) < 3:
                continue
            weight = 1.0
            if "impressions" in row and pd.notna(row["impressions"]):
                weight = max(1.0, float(row["impressions"]) / 10.0)
            evidence_rows.append({"id": f"gsc_{idx}", "source": "gsc", "text": query, "weight": weight, "metadata": json.dumps({"query": query})})
    if survey_df is not None and not survey_df.empty:
        open_text_cols = [c for c in survey_df.columns if c in ["challenges", "goals", "open_text"] or survey_df[c].dtype == object]
        for idx, row in survey_df.iterrows():
            for col in open_text_cols:
                val = row.get(col)
                if pd.notna(val):
                    text = clean_text(str(val))
                    if len(text) > 10:
                        evidence_rows.append({"id": f"survey_{idx}_{col}", "source": "survey", "text": text, "weight": 1.0, "metadata": json.dumps({"column": col})})
    if transcript_utterances:
        for idx, utt in enumerate(transcript_utterances):
            text = utt["text"]
            if len(text) > 10:
                evidence_rows.append({"id": f"transcript_{idx}", "source": "transcript", "text": text, "weight": 1.0, "metadata": json.dumps({"speaker": utt["speaker"]})})
    evidence_df = pd.DataFrame(evidence_rows)
    print(f"\n Evidence corpus built:")
    print(f"   Total evidence points: {len(evidence_df)}")
    if not evidence_df.empty:
        print(f"   By source: {evidence_df['source'].value_counts().to_dict()}")
        print(f"   Total weight: {evidence_df['weight'].sum():.1f}")
    return evidence_df

def determine_optimal_personas(evidence: pd.DataFrame, max_personas: int = 3, min_cluster_size: int = 50, min_silhouette: float = 0.15) -> int:
    """Determine optimal number of personas - only create multiple if truly differentiated"""
    if not HAS_SKLEARN:
        print(" Sklearn not available, generating single persona")
        return 1
    if len(evidence) < min_cluster_size * 2:
        print(f"ℹ Limited data ({len(evidence)} points), generating single persona")
        return 1
    print("\n Analyzing evidence for clustering...")
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
        tfidf_matrix = vectorizer.fit_transform(evidence['text'])
        silhouette_scores = []
        K_range = range(2, min(max_personas + 1, len(evidence) // min_cluster_size + 1))
        for k in K_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, labels)
            silhouette_scores.append((k, score))
            print(f"   Testing {k} clusters: silhouette score = {score:.3f}")
        if not silhouette_scores:
            return 1
        # Only use multiple personas if silhouette score is good enough
        best_k, best_score = max(silhouette_scores, key=lambda x: x[1])
        if best_score < min_silhouette:
            print(f" Clusters not well-differentiated (score {best_score:.3f} < {min_silhouette}), using single persona")
            return 1
        # Validate cluster sizes
        kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(tfidf_matrix)
        cluster_sizes = pd.Series(labels).value_counts()
        if cluster_sizes.min() < min_cluster_size:
            print(f" Some clusters too small (< {min_cluster_size}), reducing to {max(1, best_k - 1)} personas")
            best_k = max(1, best_k - 1)
        print(f" Optimal personas: {best_k} (differentiation score: {best_score:.3f})")
        return best_k
    except Exception as e:
        print(f" Clustering failed: {e}, generating single persona")
        return 1

def cluster_evidence(evidence: pd.DataFrame, n_clusters: int) -> pd.DataFrame:
    """Cluster evidence and add cluster labels"""
    if n_clusters == 1 or not HAS_SKLEARN:
        evidence['cluster'] = 0
        return evidence
    try:
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english', min_df=2)
        tfidf_matrix = vectorizer.fit_transform(evidence['text'])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        evidence['cluster'] = kmeans.fit_predict(tfidf_matrix)
        print(f" Evidence clustered into {n_clusters} groups:")
        for i in range(n_clusters):
            cluster_size = (evidence['cluster'] == i).sum()
            print(f"   Cluster {i+1}: {cluster_size} points")
        return evidence
    except Exception as e:
        print(f" Clustering failed: {e}")
        evidence['cluster'] = 0
        return evidence

ROLE_PATTERNS = {
    "SEO Manager": [r"\bseo\b", r"\bsearch\b.*\bmanager\b", r"\borganic\b.*\bmanager\b"],
    "Content Marketer": [r"\bcontent\b.*\bmarket", r"\bmarket.*\bcontent\b", r"\beditorial\b"],
    "Product Manager": [r"\bproduct\b.*\bmanag", r"\bpm\b", r"\bproduct owner\b"],
    "Founder/CEO": [r"\bfounder\b", r"\bceo\b", r"\bchief executive\b"],
    "Growth Marketer": [r"\bgrowth\b", r"\bperformance\b.*\bmarket", r"\bacquisition\b"],
    "Data Analyst": [r"\bdata\b.*\banaly", r"\banaly.*\bdata\b", r"\bbi\b"],
}

JOB_CATEGORIES = {
    "measure": [r"\bmeasure\b", r"\btrack\b", r"\bmonitor\b", r"\bmetric\b", r"\banalytics\b"],
    "optimize": [r"\boptimiz\b", r"\bimprov\b", r"\benhance\b", r"\bincrease\b"],
    "understand": [r"\bunderstand\b", r"\blearn\b", r"\bfigure out\b", r"\bknow\b"],
    "implement": [r"\bimplement\b", r"\bexecute\b", r"\broll out\b", r"\bdeploy\b"],
    "report": [r"\breport\b", r"\bshare\b", r"\bcommunicate\b", r"\bdashboard\b"],
    "stay_current": [r"\bstay\b.*\bahead\b", r"\bkeep up\b", r"\blatest\b", r"\btrend\b"],
}

CONSTRAINT_THEMES = {
    "time": [r"\btime\b", r"\bquick\b", r"\bfast\b", r"\bslow\b", r"\bhours?\b"],
    "resources": [r"\bresource\b", r"\bbudget\b", r"\bcost\b", r"\bexpensive\b", r"\bcheap\b"],
    "complexity": [r"\bcomplex\b", r"\bhard\b", r"\bdifficult\b", r"\bconfusing\b", r"\bsimple\b"],
    "uncertainty": [r"\buncertain\b", r"\bunclear\b", r"\bconfidence\b", r"\bnot sure\b"],
}

DECISION_THEMES = {
    "evidence": [r"\bevidence\b", r"\bdata\b", r"\bproof\b", r"\bcase study\b", r"\bexample\b"],
    "implementation": [r"\bhow to\b", r"\bstep\b", r"\bguide\b", r"\btutorial\b", r"\bwalkthrough\b"],
    "templates": [r"\btemplate\b", r"\bchecklist\b"],
    "depth": [r"\bdeep\b", r"\bdetail\b", r"\bcomprehensive\b", r"\bin-depth\b", r"\bprimary source\b", r"\bfirst principles\b"],
}

PAIN_THEMES = {
    "measurement_gaps": [r"\bcan't measure\b", r"\bhard to track\b", r"\bno data\b", r"\bblack box\b", r"\bvisibility\b.*\bunknown\b", r"\bimpact\b.*\bunknown\b"],
    "tool_limitations": [r"\btool\b.*\bcan't\b", r"\btool\b.*\bdoesn't\b", r"\bno tool\b"],
    "lack_guidance": [r"\bdon't know how\b", r"\bno guidance\b", r"\bconfused\b"],
    "stakeholder_pressure": [r"\bstakeholder\b", r"\bleadership\b", r"\bprove\b.*\bvalue\b", r"\bjustify\b"],
}

MOTIVATION_THEMES = {
    "fomo": [r"\bstay ahead\b", r"\bcompetitor\b", r"\bfall behind\b", r"\bmiss out\b"],
    "efficiency": [r"\bsave time\b", r"\bfaster\b", r"\bautomate\b", r"\bstreamline\b"],
    "impact": [r"\bimpact\b", r"\bresult\b", r"\boutcome\b", r"\bgrowth\b"],
    "mastery": [r"\bmaster\b", r"\bexpert\b", r"\bskill\b", r"\blearn\b"],
}

VALUE_THEMES = {
    "actionable": [r"\bactionable\b", r"\bpractical\b", r"\bapplicable\b"],
    "concise": [r"\bconcise\b", r"\bquick\b", r"\bshort\b", r"\bbite-sized\b"],
    "evidence_based": [r"\bevidence\b", r"\bdata-driven\b", r"\bresearch\b"],
}

def evidence_match_counts(evidence: pd.DataFrame, patterns: List[str]) -> Tuple[int, float]:
    if evidence.empty:
        return 0, 0.0
    combined = '|'.join(patterns)
    regex = re.compile(combined, re.IGNORECASE)
    matches = evidence['text'].str.contains(regex, na=False, regex=True)
    count = matches.sum()
    weight = evidence.loc[matches, 'weight'].sum() if count > 0 else 0.0
    return count, weight

def match_taxonomy(evidence: pd.DataFrame, taxonomy: Dict[str, List[str]], top_n: int = 3) -> List[Tuple[str, int, float]]:
    results = []
    for label, patterns in taxonomy.items():
        count, weight = evidence_match_counts(evidence, patterns)
        if count > 0:
            results.append((label, count, weight))
    results.sort(key=lambda x: (x[2], x[1]), reverse=True)
    return results[:top_n]

def extract_thematic_vocabulary(evidence: pd.DataFrame, top_n: int = 12) -> Dict[str, List[str]]:
    """Extract vocabulary grouped by themes (jobs, problems, tools)"""
    if evidence.empty or len(evidence) < 2:
        return {}

    vocab_themes = {
        "action_verbs": [],
        "tools_platforms": [],
        "metrics_outcomes": [],
        "problems_challenges": []
    }

    # Pattern-based categorization
    action_patterns = [r"\b(optimiz|improv|measur|track|analyz|implement|execut|test|build|creat|launch)\w*\b"]
    tool_patterns = [r"\b(google|analytics|seo|tool|platform|software|dashboard|template)\w*\b"]
    metric_patterns = [r"\b(traffic|rank|convers|revenue|roi|kpi|metric|growth|performance)\w*\b"]
    problem_patterns = [r"\b(difficult|challenge|problem|issue|struggle|bottleneck|limitation)\w*\b"]

    texts = ' '.join(evidence['text'].tolist()).lower()

    for pattern_list, theme in [(action_patterns, "action_verbs"), (tool_patterns, "tools_platforms"),
                                  (metric_patterns, "metrics_outcomes"), (problem_patterns, "problems_challenges")]:
        for pattern in pattern_list:
            matches = re.findall(pattern, texts)
            vocab_themes[theme].extend(matches)

    # Get top terms per theme
    result = {}
    for theme, terms in vocab_themes.items():
        if terms:
            term_counts = Counter(terms)
            result[theme] = [term for term, count in term_counts.most_common(3)]

    return result

def sample_evidence(evidence: pd.DataFrame, field: str, taxonomy_matches: List[Tuple[str, int, float]], total: int = 5, per_source: int = 2, max_chars: int = 200, exclude_ids: set = None) -> List[Dict[str, str]]:
    """Sample evidence for a field, excluding already-used evidence IDs"""
    if evidence.empty or not taxonomy_matches:
        return []
    if exclude_ids is None:
        exclude_ids = set()

    samples = []
    for label, count, weight in taxonomy_matches[:3]:
        patterns = []
        if field == "jobs":
            patterns = JOB_CATEGORIES.get(label, [])
        elif field == "constraints":
            patterns = CONSTRAINT_THEMES.get(label, [])
        elif field == "decision_criteria":
            patterns = DECISION_THEMES.get(label, [])
        elif field == "frustrations":
            patterns = PAIN_THEMES.get(label, [])
        elif field == "motivations":
            patterns = MOTIVATION_THEMES.get(label, [])
        elif field == "values":
            patterns = VALUE_THEMES.get(label, [])
        if not patterns:
            continue
        combined = '|'.join(patterns)
        regex = re.compile(combined, re.IGNORECASE)
        matches = evidence[evidence['text'].str.contains(regex, na=False, regex=True)]

        # Exclude already-used evidence
        if not matches.empty and exclude_ids:
            matches = matches[~matches['id'].isin(exclude_ids)]

        if matches.empty:
            continue
        by_source = {}
        for source in matches['source'].unique():
            source_matches = matches[matches['source'] == source]
            sampled = source_matches.nlargest(per_source, 'weight')
            by_source[source] = sampled
        for source, sampled in by_source.items():
            for idx, row in sampled.iterrows():
                if len(samples) >= total:
                    break
                samples.append({"text": truncate(row['text'], max_chars), "source": row['source'], "id": row['id']})
            if len(samples) >= total:
                break
        if len(samples) >= total:
            break
    return samples

def compute_confidence(evidence_count: int, total_evidence: int, min_threshold: int = 10) -> str:
    if evidence_count >= min_threshold and evidence_count >= total_evidence * 0.15:
        return "high"
    elif evidence_count >= min_threshold * 0.5:
        return "medium"
    else:
        return "low"

def infer_archetype(persona: Dict) -> str:
    vocab_themes = persona.get("vocabulary_themes", {})
    vocab_text = " ".join([term for terms in vocab_themes.values() for term in terms]).lower()
    jobs = [j["label"] for j in persona.get("jobs", [])]
    jobs_text = " ".join(jobs).lower()
    decision_criteria = [d["label"] for d in persona.get("decision_criteria", [])]
    decision_text = " ".join(decision_criteria).lower()
    decision_maker_score = 0
    if any(kw in vocab_text for kw in ["strategy", "budget", "roi", "leadership", "team"]):
        decision_maker_score += 2
    if any(kw in jobs_text for kw in ["report", "communicate", "share"]):
        decision_maker_score += 1
    if persona.get("role", "") and any(kw in persona["role"].lower() for kw in ["director", "head", "vp", "chief", "manager"]):
        decision_maker_score += 2
    power_user_score = 0
    if any(kw in vocab_text for kw in ["implement", "tool", "workflow", "automation", "integration"]):
        power_user_score += 2
    if any(kw in jobs_text for kw in ["implement", "optimize", "measure"]):
        power_user_score += 2
    if any(kw in decision_text for kw in ["implementation", "templates", "guide"]):
        power_user_score += 1
    skeptic_score = 0
    if any(kw in decision_text for kw in ["evidence", "proof", "case study", "data"]):
        skeptic_score += 2
    if any(kw in vocab_text for kw in ["research", "study", "analysis", "data"]):
        skeptic_score += 1
    scores = {"Decision-Maker": decision_maker_score, "Power-User": power_user_score, "Skeptic": skeptic_score}
    archetype = max(scores, key=scores.get)
    if scores[archetype] == 0:
        return "Power-User"
    return archetype

def generate_persona_name(persona: Dict, used_names: set = None) -> str:
    """Generate a unique descriptive name for the persona using multiple characteristics"""
    if used_names is None:
        used_names = set()

    archetype = persona.get("archetype", "Professional")
    jobs = [j["label"] for j in persona.get("jobs", [])]
    frustrations = [f["label"] for f in persona.get("frustrations", [])]
    decision_criteria = [d["label"] for d in persona.get("decision_criteria", [])]
    vocab_themes = persona.get("vocabulary_themes", {})

    # Extract primary characteristics
    primary_job = jobs[0] if jobs else None
    secondary_job = jobs[1] if len(jobs) > 1 else None
    primary_frustration = frustrations[0] if frustrations else None
    primary_decision = decision_criteria[0] if decision_criteria else None

    # Get top vocabulary category as another differentiator
    top_vocab_category = None
    if vocab_themes:
        max_count = 0
        for category, terms in vocab_themes.items():
            if len(terms) > max_count:
                max_count = len(terms)
                top_vocab_category = category

    # Build unique name using combination of characteristics
    name = None

    # Start with archetype-based prefix
    if archetype == "Decision-Maker":
        # Decision-Maker variations
        if primary_job == "report" and primary_frustration == "stakeholder_pressure":
            name = "The Executive Communicator"
        elif primary_job == "report":
            name = "The Strategic Reporter"
        elif primary_frustration == "stakeholder_pressure":
            name = "The ROI-Focused Leader"
        elif primary_job == "measure" and primary_decision == "roi":
            name = "The Metrics-Driven Executive"
        elif primary_job == "understand":
            name = "The Strategic Learner"
        else:
            name = "The Strategic Decision-Maker"

    elif archetype == "Skeptic":
        # Skeptic variations
        if primary_job == "measure" and primary_frustration == "measurement_gaps":
            name = "The Data Quality Analyst"
        elif primary_job == "measure":
            name = "The Data-Driven Analyst"
        elif primary_frustration == "measurement_gaps":
            name = "The Evidence-Seeking Skeptic"
        elif primary_job == "understand" and "lack_guidance" in frustrations:
            name = "The Research-First Skeptic"
        elif primary_frustration == "tool_limitations":
            name = "The Technical Evaluator"
        elif primary_job == "optimize":
            name = "The Test-Everything Optimizer"
        else:
            name = "The Research-Oriented Skeptic"

    else:  # Power-User
        # Power-User variations (most common, needs most differentiation)
        if primary_job == "implement" and secondary_job == "optimize":
            name = "The Build-and-Optimize Practitioner"
        elif primary_job == "implement" and primary_frustration == "tool_limitations":
            name = "The Technical Implementer"
        elif primary_job == "implement":
            name = "The Hands-On Implementer"
        elif primary_job == "optimize" and primary_decision == "speed":
            name = "The Rapid Optimizer"
        elif primary_job == "optimize" and top_vocab_category == "metrics":
            name = "The Performance Optimizer"
        elif primary_job == "optimize":
            name = "The Growth Optimizer"
        elif primary_job == "measure" and top_vocab_category == "tools":
            name = "The Tool-Savvy Analyst"
        elif primary_job == "measure":
            name = "The Metrics-Focused Practitioner"
        elif primary_job == "understand" and primary_frustration == "lack_guidance":
            name = "The Self-Directed Learner"
        elif primary_job == "understand":
            name = "The Continuous Learner"
        elif primary_job == "stay_current":
            name = "The Industry Tracker"
        else:
            name = "The Tactical Executor"

    # If name is already used, add a differentiator
    if name in used_names:
        # Try adding secondary characteristics
        if primary_frustration and primary_frustration != "stakeholder_pressure":
            differentiator = primary_frustration.replace("_", " ").title()
            name = f"{name} (Focused on {differentiator})"
        elif secondary_job:
            differentiator = secondary_job.replace("_", " ").title()
            name = f"{name} (Also {differentiator})"
        elif primary_decision:
            differentiator = primary_decision.replace("_", " ").title()
            name = f"{name} (Prioritizes {differentiator})"
        else:
            # Last resort: append persona ID or number
            counter = 2
            base_name = name
            while name in used_names:
                name = f"{base_name} #{counter}"
                counter += 1

    return name

def assess_data_quality(evidence: pd.DataFrame, survey_df: pd.DataFrame = None, gsc_df: pd.DataFrame = None) -> Dict:
    score = 100
    issues = []
    strengths = []
    coverage_notes = {"overrepresents": [], "underrepresents": [], "missing": []}

    total_evidence = len(evidence)
    if total_evidence < CONFIG["min_evidence_rows"]:
        issues.append({"severity": "high", "message": f"Only {total_evidence} evidence points (recommend {CONFIG['recommended_evidence_rows']}+)", "impact": -20})
        score -= 20
    elif total_evidence < CONFIG["recommended_evidence_rows"]:
        issues.append({"severity": "medium", "message": f"Only {total_evidence} evidence points (recommend {CONFIG['recommended_evidence_rows']}+)", "impact": -10})
        score -= 10
    else:
        strengths.append(f"Strong evidence base: {total_evidence} points")

    if not evidence.empty:
        source_dist = evidence['source'].value_counts(normalize=True)
        max_source_pct = source_dist.max()
        dominant_source = source_dist.idxmax()

        if max_source_pct > 0.85:
            issues.append({"severity": "medium", "message": f"Evidence heavily skewed to {dominant_source} ({max_source_pct*100:.0f}%)", "impact": -10})
            score -= 10
            coverage_notes["overrepresents"].append(f"{dominant_source} data heavily weighted")
        elif max_source_pct > 0.70:
            issues.append({"severity": "low", "message": f"Evidence somewhat skewed to {dominant_source} ({max_source_pct*100:.0f}%)", "impact": -5})
            score -= 5
            coverage_notes["overrepresents"].append(f"{dominant_source} data moderately weighted")
        else:
            strengths.append("Balanced source distribution")

        # Identify underrepresented sources
        for source, pct in source_dist.items():
            if pct < 0.1:
                coverage_notes["underrepresents"].append(f"{source} (<10% of data)")

    if not evidence.empty:
        avg_length = evidence['text'].str.len().mean()
        if avg_length < 30:
            issues.append({"severity": "medium", "message": f"Evidence snippets are short (avg {avg_length:.0f} chars)", "impact": -10})
            score -= 10
        elif avg_length >= 60:
            strengths.append(f"Good text depth: {avg_length:.0f} avg chars")

    vocab_diversity = len(set(' '.join(evidence['text'].tolist()).lower().split()))
    if vocab_diversity < 100:
        issues.append({"severity": "medium", "message": f"Limited vocabulary diversity ({vocab_diversity} unique words)", "impact": -10})
        score -= 10
    elif vocab_diversity >= 200:
        strengths.append(f"Rich vocabulary: {vocab_diversity} unique terms")

    if gsc_df is not None and 'date' in gsc_df.columns:
        try:
            gsc_df['date_parsed'] = pd.to_datetime(gsc_df['date'], errors='coerce')
            valid_dates = gsc_df['date_parsed'].dropna()
            if len(valid_dates) > 0:
                date_range = (valid_dates.max() - valid_dates.min()).days
                if date_range < 14:
                    issues.append({"severity": "medium", "message": f"Data spans only {date_range} days", "impact": -10})
                    score -= 10
                elif date_range >= 60:
                    strengths.append(f"Good temporal coverage: {date_range} days")
        except:
            pass

    # Identify completely missing data types
    if survey_df is None or survey_df.empty:
        coverage_notes["missing"].append("Survey responses (customer voice)")
    if 'transcript' not in evidence['source'].values:
        coverage_notes["missing"].append("Call transcripts (deep customer insights)")
    if 'gsc' not in evidence['source'].values:
        coverage_notes["missing"].append("Search queries (intent signals)")

    score = max(0, min(100, score))
    if score >= 80:
        recommendation = "High quality - suitable for strategic planning"
    elif score >= 60:
        recommendation = "Adequate quality for tactical content decisions"
    else:
        recommendation = "Low quality - directional insights only"

    return {
        "score": score,
        "issues": issues,
        "strengths": strengths,
        "recommendation": recommendation,
        "coverage_notes": coverage_notes
    }

def print_quality_report(quality: Dict):
    print("\n" + "="*60)
    print(" DATA QUALITY ASSESSMENT")
    print("="*60)
    score = quality["score"]
    bar_length = 40
    filled = int(bar_length * score / 100)
    bar = "" * filled + "" * (bar_length - filled)
    print(f"\n Overall Score: [{bar}] {score}/100")
    print(f"   {quality['recommendation']}")
    if quality["issues"]:
        print(f"\n {len(quality['issues'])} issue(s) detected:")
        for issue in quality["issues"]:
            severity_emoji = {"high": "", "medium": "🟡", "low": "🟢"}
            emoji = severity_emoji.get(issue["severity"], "")
            print(f"   {emoji} [{issue['severity'].upper()}] {issue['message']}")
    if quality["strengths"]:
        print(f"\n {len(quality['strengths'])} strength(s):")
        for strength in quality["strengths"]:
            print(f"   • {strength}")
    print("\n" + "="*60)

def generate_content_recommendations(persona: Dict, evidence: pd.DataFrame) -> Dict:
    recommendations = {
        "priority_content_types": [],
        "quick_wins": [],
        "messaging_angles": [],
        "specification_requirements": {}
    }

    decision_criteria = [d["label"] for d in persona.get("decision_criteria", [])]
    jobs = [j["label"] for j in persona.get("jobs", [])]
    frustrations = [f["label"] for f in persona.get("frustrations", [])]
    archetype = persona.get("archetype", "")

    # Specific content recommendations tied to persona
    primary_job = jobs[0] if jobs else "general tasks"
    primary_frustration = frustrations[0] if frustrations else "challenges"

    if "implementation" in decision_criteria or "implement" in jobs:
        topic = f"{primary_job} in your workflow"
        recommendations["priority_content_types"].append({
            "type": f"Step-by-step tutorial: How to {topic}",
            "confidence": "high",
            "rationale": f"Persona needs implementation guidance for {primary_job}"
        })

    if "evidence" in decision_criteria or archetype == "Skeptic":
        topic = primary_job if primary_job != "general tasks" else "content strategy"
        recommendations["priority_content_types"].append({
            "type": f"Case study: How [Company] improved {topic} by X%",
            "confidence": "high",
            "rationale": "Persona needs proof before taking action"
        })

    if "stakeholder_pressure" in frustrations or archetype == "Decision-Maker":
        recommendations["priority_content_types"].append({
            "type": f"ROI calculator/template for {primary_job}",
            "confidence": "high",
            "rationale": "Persona needs to justify decisions to leadership"
        })

    # Quick wins from actual questions
    if not evidence.empty:
        questions = []
        for text in evidence['text']:
            if '?' in text or any(text.lower().startswith(q) for q in ["how", "what", "why", "can", "should"]):
                questions.append(text)
        if questions:
            top_questions = questions[:5]
            recommendations["quick_wins"] = {
                "type": "FAQ-style content or Twitter threads",
                "questions": [truncate(q, 120) for q in top_questions]
            }

    # Messaging angles
    recommendations["messaging_angles"] = [
        {
            "name": "Pain-solution framing",
            "template": f"Stop {primary_frustration}. Start [solution].",
            "example": f"Stop struggling with {primary_frustration}. Start using [your framework].",
            "use_case": "Email subject lines, ad copy"
        },
        {
            "name": "Social proof",
            "template": f"How {archetype}s are solving {primary_frustration}",
            "example": f"How 3 {archetype}s transformed {primary_job}",
            "use_case": "Blog headlines, LinkedIn posts"
        },
        {
            "name": "Efficiency promise",
            "template": f"The 15-minute {primary_job} [format]",
            "example": f"The 15-minute {primary_job} checklist",
            "use_case": "Lead magnets, newsletter hooks"
        },
    ]

    # Enhanced Specification Requirements (Provenance Tracking)
    quality = assess_data_quality(evidence)
    source_counts = evidence['source'].value_counts().to_dict()
    total_evidence = len(evidence)

    spec_reqs = {
        "provenance": {
            "data_sources": {k: f"{v} points ({v/total_evidence*100:.1f}%)" for k, v in source_counts.items()},
            "date_range": "Data collected over [specify timeframe]",
            "sample_size": f"{total_evidence} total evidence points",
            "weighting": "GSC queries weighted by impressions; survey/transcripts weighted equally"
        },
        "confidence_by_field": {},
        "coverage_notes": quality.get("coverage_notes", {}),
        "validation_hooks": [],
        "update_triggers": []
    }

    # Confidence by field
    for field_name, field_data in [("jobs", persona.get("jobs", [])),
                                     ("frustrations", persona.get("frustrations", [])),
                                     ("decision_criteria", persona.get("decision_criteria", []))]:
        if field_data:
            top_item = field_data[0]
            conf = top_item.get("confidence", "low").upper()
            count = top_item.get("count", 0)
            spec_reqs["confidence_by_field"][field_name] = f"{conf} (based on {count} evidence points)"

    # Validation hooks - parity checks
    if "stakeholder_pressure" in frustrations:
        spec_reqs["validation_hooks"].append("Check if content types emphasize ROI/business impact")
    if "evidence" in decision_criteria:
        spec_reqs["validation_hooks"].append("Verify all claims are backed by data or case studies")
    if "implement" in jobs:
        spec_reqs["validation_hooks"].append("Ensure tutorials include step-by-step instructions")

    # Update triggers
    vocab_themes = persona.get("vocabulary_themes", {})
    if vocab_themes:
        spec_reqs["update_triggers"].append("Vocabulary shifts detected in new evidence (monitor quarterly)")
    spec_reqs["update_triggers"].append("New competitor content formats emerge")
    spec_reqs["update_triggers"].append(f"Change in primary {primary_job} patterns")
    if primary_frustration != "challenges":
        spec_reqs["update_triggers"].append(f"Shift in {primary_frustration} frequency or severity")

    recommendations["specification_requirements"] = spec_reqs

    return recommendations

def print_content_recommendations(recommendations: Dict):
    print("\n" + "="*60)
    print(" CONTENT STRATEGY RECOMMENDATIONS")
    print("="*60)
    print("\n Priority Content Types:")
    for i, ct in enumerate(recommendations.get("priority_content_types", []), 1):
        print(f"\n{i}. {ct['type']}")
        print(f"   Confidence: {ct['confidence']}")
        print(f"   Rationale: {ct['rationale']}")

    quick_wins = recommendations.get("quick_wins")
    if quick_wins:
        print(f"\n Quick Wins:")
        print(f"   Type: {quick_wins['type']}")
        print(f"   Top questions to answer:")
        for q in quick_wins.get("questions", [])[:3]:
            print(f'   • "{q}"')

    print(f"\n Messaging Angles:")
    for angle in recommendations.get("messaging_angles", []):
        print(f"\n   {angle['name']}:")
        print(f"   - Template: {angle['template']}")
        print(f"   - Example: {angle['example']}")
        print(f"   - Use case: {angle['use_case']}")

    spec_reqs = recommendations.get("specification_requirements", {})
    if spec_reqs:
        print(f"\n Specification Requirements (Provenance Tracking):")

        if "provenance" in spec_reqs:
            prov = spec_reqs["provenance"]
            print(f"\n   Data Sources:")
            for source, info in prov.get("data_sources", {}).items():
                print(f"     • {source}: {info}")
            print(f"   Sample Size: {prov.get('sample_size', 'N/A')}")
            print(f"   Weighting: {prov.get('weighting', 'N/A')}")

        if "confidence_by_field" in spec_reqs and spec_reqs["confidence_by_field"]:
            print(f"\n   Confidence by Field:")
            for field, conf_info in spec_reqs["confidence_by_field"].items():
                print(f"     • {field.replace('_', ' ').title()}: {conf_info}")

        if "validation_hooks" in spec_reqs and spec_reqs["validation_hooks"]:
            print(f"\n   Validation Hooks:")
            for hook in spec_reqs["validation_hooks"]:
                print(f"      {hook}")

        if "update_triggers" in spec_reqs and spec_reqs["update_triggers"]:
            print(f"\n   Update Triggers:")
            for trigger in spec_reqs["update_triggers"]:
                print(f"      {trigger}")

    print("\n" + "="*60)

def persona_from_data(evidence: pd.DataFrame, survey_df: pd.DataFrame = None, gsc_df: pd.DataFrame = None, transcript_questions: List[str] = None, config: Dict = None, persona_id: int = 1, quality: Dict = None, used_names: set = None) -> Dict:
    if config is None:
        config = CONFIG
    persona = {
        "version": "2.2",
        "generated_at": datetime.now().isoformat(),
        "persona_id": persona_id,
        "total_evidence": len(evidence),
        "data_quality": quality  # Include quality for all personas
    }

    # Role detection
    if survey_df is not None and "role" in survey_df.columns:
        role_counts = survey_df["role"].value_counts()
        top_role = role_counts.index[0] if len(role_counts) > 0 else "Unknown"
        top_role_count = role_counts.iloc[0] if len(role_counts) > 0 else 0
        total_role_responses = role_counts.sum()
        persona["role"] = top_role
        persona["role_confidence"] = compute_confidence(top_role_count, total_role_responses, 5)
        persona["role_distribution"] = {k: int(v) for k, v in role_counts.head(5).to_dict().items()}
    else:
        all_text = " ".join(evidence['text'].tolist())
        matched_roles = []
        for role, patterns in ROLE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, all_text, re.IGNORECASE):
                    matched_roles.append(role)
                    break
        if matched_roles:
            persona["role"] = matched_roles[0]
            persona["role_confidence"] = "low"
        else:
            persona["role"] = "Unknown"
            persona["role_confidence"] = "low"

    # Thematic vocabulary (improved)
    vocab_themes = extract_thematic_vocabulary(evidence, top_n=config["vocab"]["top_n"])
    persona["vocabulary_themes"] = vocab_themes

    # Track used evidence IDs across all fields to avoid duplicates
    used_evidence_ids = set()

    # Jobs with context
    jobs_matches = match_taxonomy(evidence, JOB_CATEGORIES, top_n=3)
    persona["jobs"] = []
    for label, count, weight in jobs_matches:
        job_context = INSIGHT_CONTEXT.get("jobs", {}).get(label, {})
        examples = sample_evidence(evidence, "jobs", jobs_matches, **config["examples"], exclude_ids=used_evidence_ids)
        # Track IDs of examples we used
        for ex in examples:
            used_evidence_ids.add(ex["id"])
        persona["jobs"].append({
            "label": label,
            "count": int(count),
            "weight": float(weight),
            "confidence": compute_confidence(count, len(evidence)),
            "context": job_context,
            "examples": examples
        })

    # Constraints
    constraint_matches = match_taxonomy(evidence, CONSTRAINT_THEMES, top_n=3)
    persona["constraints"] = []
    for label, count, weight in constraint_matches:
        persona["constraints"].append({
            "label": label,
            "count": int(count),
            "weight": float(weight),
            "confidence": compute_confidence(count, len(evidence))
        })

    # Decision criteria with context
    decision_matches = match_taxonomy(evidence, DECISION_THEMES, top_n=3)
    persona["decision_criteria"] = []
    for label, count, weight in decision_matches:
        decision_context = INSIGHT_CONTEXT.get("decision_criteria", {}).get(label, {})
        examples = sample_evidence(evidence, "decision_criteria", decision_matches, **config["examples"], exclude_ids=used_evidence_ids)
        # Track IDs of examples we used
        for ex in examples:
            used_evidence_ids.add(ex["id"])
        persona["decision_criteria"].append({
            "label": label,
            "count": int(count),
            "weight": float(weight),
            "confidence": compute_confidence(count, len(evidence)),
            "context": decision_context,
            "examples": examples
        })

    # Frustrations with context
    pain_matches = match_taxonomy(evidence, PAIN_THEMES, top_n=3)
    persona["frustrations"] = []
    for label, count, weight in pain_matches:
        frustration_context = INSIGHT_CONTEXT.get("frustrations", {}).get(label, {})
        examples = sample_evidence(evidence, "frustrations", pain_matches, **config["examples"], exclude_ids=used_evidence_ids)
        # Track IDs of examples we used
        for ex in examples:
            used_evidence_ids.add(ex["id"])
        persona["frustrations"].append({
            "label": label,
            "count": int(count),
            "weight": float(weight),
            "confidence": compute_confidence(count, len(evidence)),
            "context": frustration_context,
            "examples": examples
        })

    # Motivations with context
    motivation_matches = match_taxonomy(evidence, MOTIVATION_THEMES, top_n=3)
    persona["motivations"] = []
    for label, count, weight in motivation_matches:
        motivation_context = INSIGHT_CONTEXT.get("motivations", {}).get(label, {})
        persona["motivations"].append({
            "label": label,
            "count": int(count),
            "weight": float(weight),
            "confidence": compute_confidence(count, len(evidence)),
            "context": motivation_context
        })

    # Values
    value_matches = match_taxonomy(evidence, VALUE_THEMES, top_n=3)
    persona["values"] = []
    for label, count, weight in value_matches:
        persona["values"].append({
            "label": label,
            "count": int(count),
            "weight": float(weight),
            "confidence": compute_confidence(count, len(evidence))
        })

    persona["archetype"] = infer_archetype(persona)

    # Generate persona name (ensuring uniqueness across personas)
    if used_names is None:
        used_names = set()
    persona["name"] = generate_persona_name(persona, used_names)

    # Ensure all personas have all fields (even if empty) for consistency
    if not persona.get("jobs"):
        persona["jobs"] = []
    if not persona.get("frustrations"):
        persona["frustrations"] = []
    if not persona.get("motivations"):
        persona["motivations"] = []
    if not persona.get("decision_criteria"):
        persona["decision_criteria"] = []
    if not persona.get("values"):
        persona["values"] = []
    if not persona.get("constraints"):
        persona["constraints"] = []

    if transcript_questions:
        persona["common_questions"] = transcript_questions[:10]
    else:
        persona["common_questions"] = []

    source_counts = evidence['source'].value_counts().to_dict()
    persona["metadata"] = {
        "sources": {k: int(v) for k, v in source_counts.items()},
        "total_weight": float(evidence['weight'].sum())
    }

    return persona

def generate_persona_prompts(persona: Dict) -> Dict:
    """Prompts removed - users should copy the persona markdown to LLM instead

    To generate prompts for this persona:
    1. Copy the persona markdown file
    2. Paste into ChatGPT, Claude, or Gemini
    3. Ask: "Generate 25 prompts this persona would use to search for [your topic/product],
       organized by intent level (low, medium, high). Make them realistic queries
       reflecting their jobs, frustrations, and vocabulary."
    """
    return {}

def deduplicate_examples(examples: List[Dict[str, str]], max_examples: int = 3) -> List[Dict[str, str]]:
    """Deduplicate examples by text content and return up to max_examples unique items"""
    if not examples:
        return []

    seen_texts = set()
    unique_examples = []

    for ex in examples:
        text = ex.get("text", "").strip()
        # Only add if we haven't seen this exact text before
        if text and text not in seen_texts:
            seen_texts.add(text)
            unique_examples.append(ex)
            if len(unique_examples) >= max_examples:
                break

    return unique_examples

def print_persona_prompts(prompts: Dict):
    print("\n" + "="*60)
    print(" TRACKABLE PROMPTS FOR AI SEARCH MONITORING")
    print("="*60)
    print("\nPrompts organized by intent level:")
    print("  • LOW: Exploration - learning what's possible")
    print("  • MEDIUM: Evaluation - comparing solutions")
    print("  • HIGH: Decision - ready to implement\n")

    total_count = 0
    for intent_level in ["low_intent", "medium_intent", "high_intent"]:
        if intent_level in prompts:
            intent_name = intent_level.replace("_", " ").title()
            prompt_list = prompts[intent_level]
            print(f"\n{intent_name} ({len(prompt_list)} prompts):")
            for i, prompt_data in enumerate(prompt_list, 1):
                total_count += 1
                print(f"  {i}. {prompt_data['prompt']}")

    print(f"\n Total: {total_count} prompts")
    print(" These are real prompts users would type into AI search/chat")
    print(" Track them in your prompt monitoring tool")
    print("="*60)

def persona_to_markdown(persona: Dict, recommendations: Dict = None, prompts: Dict = None) -> str:
    md = f"# Subscriber Persona Card (v2.2)\n\n"
    md += f"**Persona Name:** {persona.get('name', 'Unknown')}\n\n"
    md += f"**Generated:** {persona['generated_at']}\n\n"
    md += f"**Persona ID:** {persona['persona_id']}\n\n"

    # Data Quality
    quality = persona.get("data_quality")
    if quality:
        md += f"**Data Quality Score:** {quality['score']}/100 - {quality['recommendation']}\n\n"

    md += f"## Overview\n\n"
    md += f"**Archetype:** {persona.get('archetype', 'Unknown')}\n\n"
    md += f"**Primary Role:** {persona.get('role', 'Unknown')} (confidence: {persona.get('role_confidence', 'unknown')})\n\n"

    # Vocabulary themes
    md += f"## Key Vocabulary (by Theme)\n\n"
    if persona.get("vocabulary_themes"):
        for theme, terms in persona["vocabulary_themes"].items():
            if terms:
                md += f"**{theme.replace('_', ' ').title()}:** {', '.join(terms)}\n\n"
    else:
        md += "No vocabulary themes identified.\n\n"

    # Jobs
    md += f"## Jobs-to-be-Done\n\n"
    if persona.get("jobs"):
        for i, job in enumerate(persona["jobs"], 1):
            md += f"{i}. **{job['label']}** (confidence: {job['confidence']}, count: {job['count']})\n"
            if job.get("context"):
                context = job["context"]
                md += f"   - **What:** {context.get('what', 'N/A')}\n"
                md += f"   - **Why:** {context.get('why', 'N/A')}\n"
            if job.get("examples"):
                md += f"   - **Evidence:**\n"
                unique_examples = deduplicate_examples(job["examples"], max_examples=3)
                for ex in unique_examples:
                    md += f'     - "{ex["text"]}" (source: {ex["source"]})\n'
            md += "\n"
    else:
        md += "No jobs-to-be-done identified.\n\n"

    # Frustrations
    md += f"## Frustrations\n\n"
    if persona.get("frustrations"):
        for i, frust in enumerate(persona["frustrations"], 1):
            md += f"{i}. **{frust['label']}** (confidence: {frust['confidence']}, count: {frust['count']})\n"
            if frust.get("context"):
                context = frust["context"]
                md += f"   - **What:** {context.get('what', 'N/A')}\n"
                md += f"   - **Why:** {context.get('why', 'N/A')}\n"
                md += f"   - **Impact:** {context.get('impact', 'N/A')}\n"
            if frust.get("examples"):
                md += f"   - **Evidence:**\n"
                unique_examples = deduplicate_examples(frust["examples"], max_examples=3)
                for ex in unique_examples:
                    md += f'     - "{ex["text"]}" (source: {ex["source"]})\n'
            md += "\n"
    else:
        md += "No frustrations identified.\n\n"

    # Motivations
    md += f"## Motivations\n\n"
    if persona.get("motivations"):
        for i, mot in enumerate(persona["motivations"], 1):
            md += f"{i}. **{mot['label']}** (confidence: {mot['confidence']}, count: {mot['count']})\n"
            if mot.get("context"):
                context = mot["context"]
                md += f"   - **What:** {context.get('what', 'N/A')}\n"
                md += f"   - **Why:** {context.get('why', 'N/A')}\n"
                md += f"   - **Drives:** {context.get('drives', 'N/A')}\n"
            md += "\n"
    else:
        md += "No motivations identified.\n\n"

    # Decision Criteria
    md += f"## Decision Criteria\n\n"
    if persona.get("decision_criteria"):
        for i, crit in enumerate(persona["decision_criteria"], 1):
            md += f"{i}. **{crit['label']}** (confidence: {crit['confidence']}, count: {crit['count']})\n"
            if crit.get("context"):
                context = crit["context"]
                md += f"   - **What:** {context.get('what', 'N/A')}\n"
                md += f"   - **Signals:** {context.get('signals', 'N/A')}\n"
            if crit.get("examples"):
                md += f"   - **Evidence:**\n"
                unique_examples = deduplicate_examples(crit["examples"], max_examples=3)
                for ex in unique_examples:
                    md += f'     - "{ex["text"]}" (source: {ex["source"]})\n'
            md += "\n"
    else:
        md += "No decision criteria identified.\n\n"

    # Values
    md += f"## Values\n\n"
    if persona.get("values"):
        for i, val in enumerate(persona["values"], 1):
            md += f"{i}. **{val['label']}** (confidence: {val['confidence']}, count: {val['count']})\n"
            md += "\n"
    else:
        md += "No values identified.\n\n"

    # Constraints
    md += f"## Constraints\n\n"
    if persona.get("constraints"):
        for i, const in enumerate(persona["constraints"], 1):
            md += f"{i}. **{const['label']}** (confidence: {const['confidence']}, count: {const['count']})\n"
            md += "\n"
    else:
        md += "No constraints identified.\n\n"

    # Content Strategy
    if recommendations:
        md += f"## Content Strategy\n\n"
        if recommendations.get("priority_content_types"):
            md += f"### Priority Content Types\n\n"
            for i, ct in enumerate(recommendations["priority_content_types"], 1):
                md += f"{i}. **{ct['type']}**\n"
                md += f"   - Confidence: {ct['confidence']}\n"
                md += f"   - Rationale: {ct['rationale']}\n\n"

        spec_reqs = recommendations.get("specification_requirements", {})
        if spec_reqs:
            md += f"### Provenance & Validation\n\n"
            if "provenance" in spec_reqs:
                prov = spec_reqs["provenance"]
                md += f"**Data Sources:**\n"
                for source, info in prov.get("data_sources", {}).items():
                    md += f"- {source}: {info}\n"
                md += f"\n**Sample Size:** {prov.get('sample_size', 'N/A')}\n\n"

            if "confidence_by_field" in spec_reqs:
                md += f"**Confidence by Field:**\n"
                for field, conf in spec_reqs["confidence_by_field"].items():
                    md += f"- {field.replace('_', ' ').title()}: {conf}\n"
                md += "\n"

            if "validation_hooks" in spec_reqs and spec_reqs["validation_hooks"]:
                md += f"**Validation Hooks:**\n"
                for hook in spec_reqs["validation_hooks"]:
                    md += f"- {hook}\n"
                md += "\n"

            if "update_triggers" in spec_reqs:
                md += f"**Update Triggers:**\n"
                for trigger in spec_reqs["update_triggers"]:
                    md += f"- {trigger}\n"
                md += "\n"

    # How to Generate Prompts
    md += f"## How to Generate Trackable Prompts\n\n"
    md += f"**To create prompts for this persona:**\n\n"
    md += f"1. Copy this entire persona card (this markdown file)\n"
    md += f"2. Paste into ChatGPT, Claude, or Gemini\n"
    md += f"3. Replace [CATEGORY/PROBLEM] with what you want to track:\n\n"
    md += f"**IMPORTANT:** Use the **category or problem** people search for, NOT your brand name.\n\n"
    md += f"**Examples:**\n"
    md += f'- ✓ GOOD: "SEO newsletter" (category), "how to report SEO results" (problem)\n'
    md += f'- ✗ BAD: "Growth Memo" (brand name)\n\n'
    md += f"**Why?** Track searches people make BEFORE they know you exist.\n\n"
    md += f"4. Ask the LLM:\n\n"
    md += f'```\n'
    md += f'Generate 25 realistic prompts this persona would use to search for [CATEGORY/PROBLEM], \n'
    md += f'organized by intent level:\n'
    md += f'- LOW intent (7 prompts): Exploration - learning what\'s possible\n'
    md += f'- MEDIUM intent (9 prompts): Evaluation - comparing solutions\n'
    md += f'- HIGH intent (9 prompts): Decision - ready to implement\n\n'
    md += f'Make them realistic queries reflecting this persona\'s jobs-to-be-done, frustrations, \n'
    md += f'decision criteria, and vocabulary. DO NOT include the category/problem name in every prompt.\n'
    md += f'```\n\n'
    md += f"5. Track the generated prompts in your AI search monitoring tool\n\n"
    md += f"**Suggested categories/problems to track:**\n"
    md += f"- If you're a newsletter: 'SEO newsletter', 'growth marketing insights', 'tactical SEO guide'\n"
    md += f"- If you're a tool: 'SEO analytics', 'content performance tracking', 'AI search monitoring'\n"
    md += f"- If you're educational: 'how to prove SEO value', 'SEO reporting frameworks', 'stakeholder communication'\n\n"

    md += f"## Metadata\n\n"
    md += f"- Total evidence points: {persona['total_evidence']}\n"
    if persona.get("metadata"):
        md += f"- Sources: {persona['metadata'].get('sources', {})}\n"
        md += f"- Total weight: {persona['metadata'].get('total_weight', 0):.1f}\n"

    return md

def persona_to_html(persona: Dict, recommendations: Dict = None, prompts: Dict = None) -> str:
    html = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>Subscriber Persona Card v2.1</title>
<style>
body {font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Arial, sans-serif; line-height: 1.6; max-width: 1000px; margin: 40px auto; padding: 20px; background: #f5f5f5;}
.card {background: white; border-radius: 8px; padding: 30px; margin-bottom: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);}
h1 {color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px;}
h2 {color: #555; margin-top: 30px; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px;}
h3 {color: #666; margin-top: 20px;}
.badge {display: inline-block; padding: 4px 12px; border-radius: 12px; font-size: 12px; font-weight: bold; margin-left: 8px;}
.badge-high {background: #4CAF50; color: white;}
.badge-medium {background: #FFC107; color: #333;}
.badge-low {background: #F44336; color: white;}
.field-item {margin: 15px 0; padding: 15px; background: #f9f9f9; border-left: 4px solid #4CAF50; border-radius: 4px;}
.context-box {background: #E3F2FD; padding: 12px; margin: 10px 0; border-radius: 4px; font-size: 14px; line-height: 1.6;}
.vocab-section {margin: 15px 0;}
.vocab-theme {background: #f0f0f0; padding: 10px; margin: 5px 0; border-radius: 4px;}
.quality-score {font-size: 48px; font-weight: bold; color: #4CAF50; text-align: center; margin: 20px 0;}
.spec-section {background: #FFF3E0; padding: 15px; margin: 10px 0; border-left: 4px solid #FF9800; border-radius: 4px;}
</style>
</head>
<body>
"""
    html += '<div class="card">'
    html += '<h1>Subscriber Persona Card v2.1</h1>'
    html += f'<h2 style="color: #4CAF50; border: none; margin-top: 10px;">{persona.get("name", "Unknown Persona")}</h2>'
    html += f'<p><strong>Generated:</strong> {persona["generated_at"][:10]}</p>'
    html += f'<p><strong>Persona ID:</strong> {persona["persona_id"]}</p>'

    quality = persona.get("data_quality")
    if quality:
        html += f'<div class="quality-score">{quality["score"]}/100</div>'
        html += f'<p style="text-align: center;">{quality["recommendation"]}</p>'

    html += f'<p><strong>Archetype:</strong> {persona.get("archetype", "Unknown")}</p>'
    html += f'<p><strong>Primary Role:</strong> {persona.get("role", "Unknown")} '
    confidence = persona.get("role_confidence", "low")
    badge_class = f"badge-{confidence}"
    html += f'<span class="badge {badge_class}">{confidence.upper()}</span></p>'
    html += '</div>'

    # Vocabulary themes
    if persona.get("vocabulary_themes"):
        html += '<div class="card">'
        html += '<h2>Key Vocabulary (by Theme)</h2>'
        html += '<div class="vocab-section">'
        for theme, terms in persona["vocabulary_themes"].items():
            if terms:
                html += f'<div class="vocab-theme">'
                html += f'<strong>{theme.replace("_", " ").title()}:</strong> {", ".join(terms)}'
                html += '</div>'
        html += '</div>'
        html += '</div>'

    # Jobs
    if persona.get("jobs"):
        html += '<div class="card">'
        html += '<h2>Jobs-to-be-Done</h2>'
        for job in persona["jobs"]:
            conf = job.get("confidence", "low")
            badge_class = f"badge-{conf}"
            html += f'<div class="field-item">'
            html += f'<strong>{job["label"]}</strong> <span class="badge {badge_class}">{conf.upper()}</span>'
            html += f'<br><small>Count: {job["count"]}</small>'
            if job.get("context"):
                context = job["context"]
                html += '<div class="context-box">'
                html += f'<strong>What:</strong> {context.get("what", "N/A")}<br>'
                html += f'<strong>Why:</strong> {context.get("why", "N/A")}'
                html += '</div>'
            if job.get("examples"):
                html += '<div style="margin-top: 10px;"><strong>Evidence:</strong><ul style="margin: 5px 0; padding-left: 20px;">'
                unique_examples = deduplicate_examples(job["examples"], max_examples=3)
                for ex in unique_examples:
                    html += f'<li>"{ex["text"]}" <em>(source: {ex["source"]})</em></li>'
                html += '</ul></div>'
            html += '</div>'
        html += '</div>'

    # Frustrations
    if persona.get("frustrations"):
        html += '<div class="card">'
        html += '<h2>Frustrations</h2>'
        for frust in persona["frustrations"]:
            conf = frust.get("confidence", "low")
            badge_class = f"badge-{conf}"
            html += f'<div class="field-item">'
            html += f'<strong>{frust["label"]}</strong> <span class="badge {badge_class}">{conf.upper()}</span>'
            html += f'<br><small>Count: {frust["count"]}</small>'
            if frust.get("context"):
                context = frust["context"]
                html += '<div class="context-box">'
                html += f'<strong>What:</strong> {context.get("what", "N/A")}<br>'
                html += f'<strong>Why:</strong> {context.get("why", "N/A")}<br>'
                html += f'<strong>Impact:</strong> {context.get("impact", "N/A")}'
                html += '</div>'
            if frust.get("examples"):
                html += '<div style="margin-top: 10px;"><strong>Evidence:</strong><ul style="margin: 5px 0; padding-left: 20px;">'
                unique_examples = deduplicate_examples(frust["examples"], max_examples=3)
                for ex in unique_examples:
                    html += f'<li>"{ex["text"]}" <em>(source: {ex["source"]})</em></li>'
                html += '</ul></div>'
            html += '</div>'
        html += '</div>'

    # Motivations
    if persona.get("motivations"):
        html += '<div class="card">'
        html += '<h2>Motivations</h2>'
        for mot in persona["motivations"]:
            conf = mot.get("confidence", "low")
            badge_class = f"badge-{conf}"
            html += f'<div class="field-item">'
            html += f'<strong>{mot["label"]}</strong> <span class="badge {badge_class}">{conf.upper()}</span>'
            if mot.get("context"):
                context = mot["context"]
                html += '<div class="context-box">'
                html += f'<strong>What:</strong> {context.get("what", "N/A")}<br>'
                html += f'<strong>Why:</strong> {context.get("why", "N/A")}<br>'
                html += f'<strong>Drives:</strong> {context.get("drives", "N/A")}'
                html += '</div>'
            html += '</div>'
        html += '</div>'

    # Decision Criteria
    if persona.get("decision_criteria"):
        html += '<div class="card">'
        html += '<h2>Decision Criteria</h2>'
        for crit in persona["decision_criteria"]:
            conf = crit.get("confidence", "low")
            badge_class = f"badge-{conf}"
            html += f'<div class="field-item">'
            html += f'<strong>{crit["label"]}</strong> <span class="badge {badge_class}">{conf.upper()}</span>'
            if crit.get("context"):
                context = crit["context"]
                html += '<div class="context-box">'
                html += f'<strong>What:</strong> {context.get("what", "N/A")}<br>'
                html += f'<strong>Signals:</strong> {context.get("signals", "N/A")}'
                html += '</div>'
            if crit.get("examples"):
                html += '<div style="margin-top: 10px;"><strong>Evidence:</strong><ul style="margin: 5px 0; padding-left: 20px;">'
                unique_examples = deduplicate_examples(crit["examples"], max_examples=3)
                for ex in unique_examples:
                    html += f'<li>"{ex["text"]}" <em>(source: {ex["source"]})</em></li>'
                html += '</ul></div>'
            html += '</div>'
        html += '</div>'

    # Content Strategy
    if recommendations:
        html += '<div class="card">'
        html += '<h2>Content Strategy</h2>'

        if recommendations.get("priority_content_types"):
            html += '<h3>Priority Content Types</h3>'
            for ct in recommendations["priority_content_types"]:
                html += f'<div class="field-item">'
                html += f'<strong>{ct["type"]}</strong><br>'
                html += f'<small>Confidence: {ct["confidence"]} | {ct["rationale"]}</small>'
                html += '</div>'

        spec_reqs = recommendations.get("specification_requirements", {})
        if spec_reqs:
            html += '<h3>Provenance & Validation</h3>'
            html += '<div class="spec-section">'

            if "provenance" in spec_reqs:
                prov = spec_reqs["provenance"]
                html += '<strong>Data Sources:</strong><ul>'
                for source, info in prov.get("data_sources", {}).items():
                    html += f'<li>{source}: {info}</li>'
                html += '</ul>'
                html += f'<strong>Sample Size:</strong> {prov.get("sample_size", "N/A")}'

            html += '</div>'

        html += '</div>'

    # How to Generate Prompts
    html += '<div class="card">'
    html += '<h2>How to Generate Trackable Prompts</h2>'
    html += '<div style="background: #FFF3E0; padding: 15px; border-radius: 4px; margin-bottom: 20px; border-left: 4px solid #FF9800;">'
    html += '<strong>⚠️ IMPORTANT:</strong> Use the <strong>category or problem</strong> people search for, NOT your brand name.<br><br>'
    html += '✓ GOOD: "SEO newsletter", "how to report SEO results"<br>'
    html += '✗ BAD: "Growth Memo"<br><br>'
    html += '<strong>Why?</strong> Track searches people make BEFORE they know you exist.'
    html += '</div>'
    html += '<p style="margin-bottom: 15px;">To create prompts for this persona:</p>'
    html += '<ol style="margin: 15px 0; line-height: 1.8;">'
    html += '<li>Copy the persona markdown file (persona_X.md)</li>'
    html += '<li>Paste into ChatGPT, Claude, or Gemini</li>'
    html += '<li>Replace [CATEGORY/PROBLEM] with what you want to track</li>'
    html += '<li>Ask the LLM to generate prompts using this template:</li>'
    html += '</ol>'
    html += '<div style="background: #f5f5f5; padding: 20px; border-radius: 4px; margin: 20px 0; border-left: 4px solid #4CAF50;">'
    html += '<pre style="margin: 0; white-space: pre-wrap; font-family: monospace; font-size: 13px; line-height: 1.6;">Generate 25 realistic prompts this persona would use to search for [CATEGORY/PROBLEM],\norganized by intent level:\n\n- LOW intent (7 prompts): Exploration - learning what\'s possible\n- MEDIUM intent (9 prompts): Evaluation - comparing solutions  \n- HIGH intent (9 prompts): Decision - ready to implement\n\nMake them realistic queries reflecting this persona\'s jobs-to-be-done, frustrations,\ndecision criteria, and vocabulary. DO NOT include the category/problem name in every prompt.</pre>'
    html += '</div>'
    html += '<ol start="5" style="margin: 15px 0; line-height: 1.8;">'
    html += '<li>Track the generated prompts in your AI search monitoring tool</li>'
    html += '</ol>'
    html += '<div style="background: #E3F2FD; padding: 15px; border-radius: 4px; margin-top: 20px;">'
    html += '<strong>Suggested categories/problems to track:</strong><ul style="margin: 10px 0;">'
    html += '<li>If you\'re a newsletter: "SEO newsletter", "growth marketing insights", "tactical SEO guide"</li>'
    html += '<li>If you\'re a tool: "SEO analytics", "content performance tracking", "AI search monitoring"</li>'
    html += '<li>If you\'re educational: "how to prove SEO value", "SEO reporting frameworks", "stakeholder communication"</li>'
    html += '</ul></div>'
    html += '</div>'

    html += '</body></html>'
    return html

def main():
    print("="*60)
    print(" SYNTHETIC PERSONA GENERATOR v2.1")
    print("="*60)
    print("\n NEW in v2.1:")
    print("  • Manual file selection (no auto-download)")
    print("  • Enhanced provenance tracking")
    print("  • Smarter persona clustering")
    print("  • Specific content strategy")
    print("  • Consistent persona structure")
    print("  • Improved thematic vocabulary")
    print("\nThis tool generates data-driven subscriber personas from:")
    print("  • Google Search Console queries")
    print("  • Survey responses")
    print("  • Call transcripts")
    print("\n" + "="*60)

    # File uploads
    print("\n STEP 1: Upload Data Files")
    print("-" * 60)
    gsc_file = upload_file(" Upload GSC CSV export (required):", required=True)
    if not gsc_file:
        print(" GSC file is required. Exiting.")
        return
    survey_file = upload_file(" Upload survey CSV (optional):", required=False)
    transcript_files = upload_multiple_files(" Upload transcript files (optional, .txt or .zip):")

    # Configure filters
    print("\n STEP 2: Configure Filters")
    print("-" * 60)
    if transcript_files:
        speaker_input = input("Filter transcripts by speaker name (comma-separated, or press Enter for all): ").strip()
        if speaker_input:
            CONFIG["speaker_filters"] = [s.strip() for s in speaker_input.split(",")]
            print(f"    Will filter for speakers: {CONFIG['speaker_filters']}")
        else:
            print("    Using all speakers")

    # Load and parse
    print("\n STEP 3: Load and Parse Data")
    print("-" * 60)
    print("Loading GSC data...")
    gsc_df, gsc_mapping = load_gsc_csv(gsc_file["content"])
    survey_df = None
    if survey_file:
        print("\nLoading survey data...")
        survey_df, survey_mapping = load_survey_csv(survey_file["content"])
    transcript_utterances = []
    transcript_questions = []
    if transcript_files:
        print("\nParsing transcripts...")
        transcript_utterances, transcript_questions = load_transcripts(transcript_files, CONFIG["speaker_filters"])

    # Build evidence
    print("\n STEP 4: Build Evidence Corpus")
    print("-" * 60)
    evidence = build_evidence_corpus(gsc_df, survey_df, transcript_utterances)
    if evidence.empty:
        print(" No evidence collected. Cannot generate persona.")
        return

    # Assess quality
    print("\n STEP 5: Assess Data Quality")
    print("-" * 60)
    quality = assess_data_quality(evidence, survey_df, gsc_df)
    print_quality_report(quality)
    if quality["score"] < 60:
        proceed = input("\n Low data quality score. Proceed anyway? (y/n): ").strip().lower()
        if proceed != 'y':
            print(" User cancelled due to data quality concerns.")
            return

    # Determine personas
    print("\n STEP 6: Determine Persona Count")
    print("-" * 60)
    n_personas = determine_optimal_personas(evidence, CONFIG["max_personas"], CONFIG["min_cluster_size"], CONFIG["min_silhouette_score"])

    if n_personas > 1:
        evidence = cluster_evidence(evidence, n_personas)
    else:
        evidence['cluster'] = 0

    # Generate personas
    print(f"\n STEP 7: Generate {n_personas} Persona(s)")
    print("-" * 60)
    personas = []
    all_recommendations = []
    all_prompts = []
    used_names = set()  # Track persona names to ensure uniqueness

    for i in range(n_personas):
        persona_evidence = evidence[evidence['cluster'] == i].copy()
        print(f"\nGenerating Persona {i+1} ({len(persona_evidence)} evidence points)...")
        persona = persona_from_data(persona_evidence, survey_df, gsc_df, transcript_questions, CONFIG, persona_id=i+1, quality=quality, used_names=used_names)

        # Validate persona has enough content
        has_jobs = len(persona.get("jobs", [])) > 0
        has_frustrations = len(persona.get("frustrations", [])) > 0
        has_sufficient_evidence = persona.get("total_evidence", 0) >= 100

        if not (has_jobs or has_frustrations) or not has_sufficient_evidence:
            print(f" Skipping Persona {i+1}: Insufficient data (jobs: {len(persona.get('jobs', []))}, frustrations: {len(persona.get('frustrations', []))}, evidence: {persona.get('total_evidence', 0)})")
            continue

        personas.append(persona)
        # Track the name we just used
        used_names.add(persona["name"])
        print(f" Persona {i+1} generated:")
        print(f"   Archetype: {persona['archetype']}")
        print(f"   Name: {persona['name']}")
        print(f"   Role: {persona.get('role', 'Unknown')}")
        print(f"   Evidence points: {persona['total_evidence']}")

        print(f"\n Generating Content Strategy for Persona {i+1}...")
        recommendations = generate_content_recommendations(persona, persona_evidence)
        all_recommendations.append(recommendations)

        # Prompts removed - users will copy markdown to LLM instead
        all_prompts.append({})

    # Summary
    if n_personas > 1:
        print("\n" + "="*60)
        print(" MULTI-PERSONA SUMMARY")
        print("="*60)
        for i, persona in enumerate(personas, 1):
            print(f"\nPersona {i}:")
            print(f"  • Archetype: {persona['archetype']}")
            print(f"  • Role: {persona.get('role', 'Unknown')}")
            print(f"  • Evidence: {persona['total_evidence']} points")
            print(f"  • Top job: {persona['jobs'][0]['label'] if persona.get('jobs') else 'N/A'}")
        print("\n" + "="*60)

    # Show recommendations
    print("\n Content Strategy (Persona 1):")
    print_content_recommendations(all_recommendations[0])

    # Generate files
    print("\n STEP 8: Generate Outputs")
    print("-" * 60)
    output_dir = "/tmp"
    os.makedirs(output_dir, exist_ok=True)

    generated_files = []

    for i, persona in enumerate(personas, 1):
        suffix = f"_persona{i}" if n_personas > 1 else ""

        # Markdown
        markdown_output = persona_to_markdown(persona, all_recommendations[i-1], all_prompts[i-1])
        md_filename = f"persona_card{suffix}.md"
        with open(os.path.join(output_dir, md_filename), "w", encoding='utf-8') as f:
            f.write(markdown_output)
        generated_files.append(md_filename)
        print(f"   Markdown persona card {i}")

        # JSON
        json_output = {
            "persona": persona,
            "recommendations": all_recommendations[i-1],
            "prompts": all_prompts[i-1]
        }
        json_filename = f"persona_card{suffix}.json"
        with open(os.path.join(output_dir, json_filename), "w", encoding='utf-8') as f:
            json.dump(json_output, f, indent=2)
        generated_files.append(json_filename)
        print(f"   JSON persona card {i}")

        # HTML
        if CONFIG["generate_html"]:
            html_output = persona_to_html(persona, all_recommendations[i-1], all_prompts[i-1])
            html_filename = f"persona_card{suffix}.html"
            with open(os.path.join(output_dir, html_filename), "w", encoding='utf-8') as f:
                f.write(html_output)
            generated_files.append(html_filename)
            print(f"   HTML report {i}")

    # Evidence CSV
    evidence.to_csv(os.path.join(output_dir, "evidence.csv"), index=False)
    generated_files.append("evidence.csv")
    print("    Evidence CSV (with cluster labels)")

    # Run summary
    run_summary = {
        "generated_at": datetime.now().isoformat(),
        "version": "2.2",
        "config": CONFIG,
        "n_personas": n_personas,
        "data_sources": {
            "gsc_rows": len(gsc_df),
            "survey_rows": len(survey_df) if survey_df is not None else 0,
            "transcript_utterances": len(transcript_utterances),
            "transcript_questions": len(transcript_questions)
        },
        "evidence": {
            "total_points": len(evidence),
            "by_source": evidence['source'].value_counts().to_dict() if not evidence.empty else {},
            "by_cluster": evidence['cluster'].value_counts().to_dict() if 'cluster' in evidence.columns else {}
        },
        "quality": quality
    }
    with open(os.path.join(output_dir, "run_summary.json"), "w", encoding='utf-8') as f:
        json.dump(run_summary, f, indent=2)
    generated_files.append("run_summary.json")
    print("    Run summary")

    # Manual file selection
    if IN_COLAB:
        print("\n STEP 9: Download Files")
        print("-" * 60)
        print("\n Generated files:")
        for i, filename in enumerate(generated_files, 1):
            print(f"   {i}. {filename}")

        print("\n Options:")
        print("   • Type 'all' to download all files")
        print("   • Type file numbers separated by commas (e.g., '1,2,3')")
        print("   • Type 'skip' to skip downloads")

        selection = input("\nYour choice: ").strip().lower()

        if selection == 'all':
            for filename in generated_files:
                filepath = os.path.join(output_dir, filename)
                files.download(filepath)
                print(f"    Downloaded: {filename}")
        elif selection != 'skip':
            try:
                indices = [int(x.strip()) - 1 for x in selection.split(',')]
                for idx in indices:
                    if 0 <= idx < len(generated_files):
                        filename = generated_files[idx]
                        filepath = os.path.join(output_dir, filename)
                        files.download(filepath)
                        print(f"    Downloaded: {filename}")
            except:
                print(" Invalid selection. Files saved to /tmp but not downloaded.")
        else:
            print("⏭ Skipped downloads. Files saved to /tmp")
    else:
        print(f"\n Files saved to: {output_dir}")

    # Final summary
    print("\n" + "="*60)
    print(" Complete!")
    print("="*60)
    print("\n Results Summary:")
    print(f"   • Personas Generated: {n_personas}")
    print(f"   • Data Quality: {quality['score']}/100")
    print(f"   • Total Evidence Points: {len(evidence)}")
    print(f"   • Output Location: {output_dir}")
    print("\n Generated Files:")
    for filename in generated_files:
        print(f"   • {filename}")
    print("\n Next Steps:")
    print("   1. Review persona cards with enhanced provenance tracking")
    print("   2. Use LLM prompts (in JSON) to test content")
    print("   3. Apply specific content strategy recommendations")
    print("   4. Monitor update triggers for persona refresh")
    print("\n" + "="*60)
    print("Thank you for using Synthetic Persona Generator v2.1! ")
    print("="*60)

if __name__ == "__main__":
    main()

