"""
Forum scraping and lay language mapping utilities.
Provides mock implementation for Reddit/Patient.info scraping and lay-to-medical terminology mapping.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ForumPost:
    """Represents a scraped forum post."""
    id: str
    title: str
    content: str
    timestamp: str
    forum_source: str
    lay_terms: List[str]
    extracted_symptoms: List[str]
    confidence_score: float


class ForumScraper:
    """
    Scrapes health forums for lay language examples.
    Mock implementation that generates realistic synthetic forum posts.
    """
    
    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize forum scraper.
        
        Args:
            cache_dir: Directory to cache scraped posts
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("forum_data")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Mock forum post templates
        self._post_templates = [
            "I've been {symptom} for {time} and I'm really worried. {detail}",
            "Does anyone else get {symptom}? It's been {time} and getting worse.",
            "Help! {symptom} and {other_symptom}. Started {time} ago.",
            "My {relation} has {symptom}. Should we go to ER? {detail}",
            "Been {symptom} since {time}. Doctor says {diagnosis} but not sure."
        ]
        
        self._details = [
            "Can't sleep at night.",
            "It's affecting my daily life.",
            "Never had this before.",
            "Getting really scared.",
            "What should I do?"
        ]
    
    def scrape_reddit_health(self, max_posts: int = 10) -> List[ForumPost]:
        """
        Scrape health-related posts from Reddit (mock implementation).
        
        Args:
            max_posts: Maximum number of posts to scrape
            
        Returns:
            List of ForumPost objects
        """
        posts = []
        
        # Condition-specific symptom mappings for realistic forum posts
        condition_symptoms = {
            "asthma": {
                "primary": [
                    ("can't breathe", "dyspnea"),
                    ("wheezy", "wheezing"),
                    ("tight chest", "chest_tightness"),
                ],
                "secondary": [
                    ("coughing", "cough"),
                    ("really tired", "fatigue"),
                    ("waking up at night", "nocturnal_symptoms")
                ]
            },
            "pneumonia": {
                "primary": [
                    ("coughing up stuff", "productive_cough"),
                    ("chest hurts", "chest_pain"),
                    ("burning up", "fever"),
                ],
                "secondary": [
                    ("can't breathe", "dyspnea"),
                    ("really tired", "fatigue"),
                    ("aching all over", "myalgia")
                ]
            },
            "copd": {
                "primary": [
                    ("can't catch my breath", "dyspnea"),
                    ("coughing up stuff", "productive_cough"),
                    ("wheezy", "wheezing"),
                ],
                "secondary": [
                    ("really tired", "fatigue"),
                    ("tight chest", "chest_tightness"),
                    ("waking up short of breath", "orthopnea")
                ]
            },
            "bronchitis": {
                "primary": [
                    ("hacking cough", "persistent_cough"),
                    ("coughing up stuff", "productive_cough"),
                    ("chest hurts from coughing", "chest_pain"),
                ],
                "secondary": [
                    ("wheezy", "wheezing"),
                    ("really tired", "fatigue"),
                    ("stuffy nose", "nasal_congestion")
                ]
            }
        }
        
        # Demographics for variation
        age_groups = ["young adult", "middle-aged", "elderly", "child"]
        durations = ["2 days", "a week", "3 hours", "yesterday", "this morning", "several weeks"]
        severities = ["mild", "getting worse", "really bad", "unbearable"]
        relations = ["kid", "mom", "dad", "partner", "friend", "grandparent"]
        
        conditions = list(condition_symptoms.keys())
        
        for i in range(max_posts):
            # Choose condition and select appropriate symptoms
            condition = random.choice(conditions)
            symptoms_data = condition_symptoms[condition]
            
            # Select 2-4 symptoms (mostly primary, some secondary)
            num_symptoms = random.randint(2, 4)
            selected_symptoms = []
            
            # Always include at least one primary symptom
            primary_symptom = random.choice(symptoms_data["primary"])
            selected_symptoms.append(primary_symptom)
            
            # Add more symptoms
            for _ in range(num_symptoms - 1):
                if random.random() < 0.7:  # 70% chance of primary symptom
                    available = [s for s in symptoms_data["primary"] if s not in selected_symptoms]
                    if available:
                        selected_symptoms.append(random.choice(available))
                    else:
                        available = [s for s in symptoms_data["secondary"] if s not in selected_symptoms]
                        if available:
                            selected_symptoms.append(random.choice(available))
                else:
                    available = [s for s in symptoms_data["secondary"] if s not in selected_symptoms]
                    if available:
                        selected_symptoms.append(random.choice(available))
            
            # Extract lay terms and medical terms
            lay_terms = [s[0] for s in selected_symptoms]
            medical_terms = [s[1] for s in selected_symptoms]
            
            # Build content with variation
            main_symptom = lay_terms[0]
            other_symptoms = lay_terms[1:] if len(lay_terms) > 1 else []
            
            # Choose template based on number of symptoms
            if len(other_symptoms) == 0:
                template = random.choice([
                    "I've been {symptom} for {time} and I'm really worried. {detail}",
                    "Does anyone else get {symptom}? It's been {time} and {severity}.",
                    "My {relation} has {symptom}. Should we go to ER? {detail}"
                ])
                content = template.format(
                    symptom=main_symptom,
                    time=random.choice(durations),
                    detail=random.choice(self._details),
                    severity=random.choice(severities),
                    relation=random.choice(relations)
                )
            elif len(other_symptoms) == 1:
                template = random.choice([
                    "Help! {symptom} and {other_symptom}. Started {time} ago.",
                    "I've been {symptom} and also {other_symptom} for {time}. {detail}",
                    "Been {symptom} since {time}, now {other_symptom} too. Getting {severity}."
                ])
                content = template.format(
                    symptom=main_symptom,
                    other_symptom=other_symptoms[0],
                    time=random.choice(durations),
                    detail=random.choice(self._details),
                    severity=random.choice(severities)
                )
            else:
                # Multiple symptoms
                symptoms_list = ", ".join(other_symptoms[:-1]) + " and " + other_symptoms[-1]
                template = random.choice([
                    "I've been {symptom}, plus {other_symptoms}. Started {time} ago. {detail}",
                    "{symptom} along with {other_symptoms}. This has been going on for {time}. {detail}",
                    "Having {symptom}, {other_symptoms}. {time} now and {severity}."
                ])
                content = template.format(
                    symptom=main_symptom,
                    other_symptoms=symptoms_list,
                    time=random.choice(durations),
                    detail=random.choice(self._details),
                    severity=random.choice(severities)
                )
            
            # Add optional demographic hint
            if random.random() < 0.3:  # 30% chance of adding age/context
                age_context = [
                    f"I'm {random.choice(age_groups)}.",
                    f"Never had this before.",
                    f"This is the worst it's been."
                ]
                content += " " + random.choice(age_context)
            
            post = ForumPost(
                id=f"reddit_{i}",
                title=f"Question about {main_symptom}",
                content=content,
                timestamp=datetime.now().isoformat(),
                forum_source="reddit",
                lay_terms=lay_terms,
                extracted_symptoms=medical_terms,
                confidence_score=random.uniform(0.7, 0.95)
            )
            posts.append(post)
        
        return posts
    
    def scrape_patient_info(self, max_posts: int = 10) -> List[ForumPost]:
        """
        Scrape posts from Patient.info forums (mock implementation).
        
        Args:
            max_posts: Maximum number of posts to scrape
            
        Returns:
            List of ForumPost objects
        """
        # Similar to Reddit but with slightly different phrasing
        posts = self.scrape_reddit_health(max_posts)
        for post in posts:
            post.forum_source = "patient.info"
            post.id = post.id.replace("reddit", "patientinfo")
        return posts
    
    def save_posts(self, posts: List[ForumPost], filename: str) -> None:
        """
        Save posts to JSON file.
        
        Args:
            posts: List of ForumPost objects
            filename: Output filename
        """
        filepath = self.cache_dir / filename
        with open(filepath, 'w') as f:
            json.dump([asdict(post) for post in posts], f, indent=2)
    
    def load_posts(self, filename: str) -> List[ForumPost]:
        """
        Load posts from JSON file.
        
        Args:
            filename: Input filename
            
        Returns:
            List of ForumPost objects
        """
        filepath = self.cache_dir / filename
        with open(filepath, 'r') as f:
            data = json.load(f)
        return [ForumPost(**post) for post in data]


class LayLanguageMapper:
    """
    Maps between medical terminology and lay language.
    """
    
    def __init__(self):
        """Initialize with predefined mappings."""
        # Predefined lay-to-medical mappings
        self.mappings = {
            "can't breathe": "dyspnea",
            "short of breath": "dyspnea",
            "breathless": "dyspnea",
            "wheezy": "wheezing",
            "whistling breath": "wheezing",
            "tight chest": "chest_tightness",
            "chest feels tight": "chest_tightness",
            "coughing up stuff": "productive_cough",
            "bringing up phlegm": "sputum_production",
            "chest hurts": "chest_pain",
            "burning chest": "chest_pain",
            "really tired": "fatigue",
            "wiped out": "fatigue",
            "exhausted": "fatigue",
            "stuffy nose": "nasal_congestion",
            "blocked nose": "nasal_congestion",
            "scratchy throat": "sore_throat",
            "throat hurts": "sore_throat",
            "can't catch my breath": "dyspnea",
            "gasping for air": "severe_dyspnea",
            "drowning feeling": "orthopnea",
            "hacking cough": "persistent_cough",
            "dry cough": "dry_cough",
            "hot": "fever",
            "burning up": "fever",
            "aching all over": "myalgia"
        }
        
        # Build reverse mapping
        self._build_reverse_mapping()
    
    def _build_reverse_mapping(self) -> None:
        """Build medical-to-lay reverse mapping."""
        self.reverse_mappings = {}
        for lay, medical in self.mappings.items():
            if medical not in self.reverse_mappings:
                self.reverse_mappings[medical] = []
            self.reverse_mappings[medical].append(lay)
    
    def get_medical_term(self, lay_term: str) -> Optional[str]:
        """
        Get medical term for a lay term.
        
        Args:
            lay_term: Lay language term
            
        Returns:
            Medical term or None if not found
        """
        return self.mappings.get(lay_term.lower())
    
    def get_lay_terms_for_medical(self, medical_term: str) -> List[str]:
        """
        Get lay terms for a medical term.
        
        Args:
            medical_term: Medical terminology
            
        Returns:
            List of lay terms
        """
        return self.reverse_mappings.get(medical_term, [])
    
    def update_mappings_from_posts(self, posts: List[ForumPost]) -> None:
        """
        Update mappings based on forum posts.
        
        Args:
            posts: List of forum posts with extracted terms
        """
        for post in posts:
            for lay_term, medical_term in zip(post.lay_terms, post.extracted_symptoms):
                if lay_term.lower() not in self.mappings:
                    self.mappings[lay_term.lower()] = medical_term
        
        # Rebuild reverse mapping
        self._build_reverse_mapping()
    
    def save_mappings(self, filepath: str) -> None:
        """
        Save mappings to JSON file.
        
        Args:
            filepath: Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(self.mappings, f, indent=2)
    
    def load_mappings(self, filepath: str) -> None:
        """
        Load mappings from JSON file.
        
        Args:
            filepath: Input file path
        """
        with open(filepath, 'r') as f:
            self.mappings = json.load(f)
        self._build_reverse_mapping()


class ForumDataAugmentation:
    """
    Augments medical complaints with forum-derived lay language.
    """
    
    def __init__(self, mapper: Optional[LayLanguageMapper] = None):
        """
        Initialize data augmentation.
        
        Args:
            mapper: LayLanguageMapper instance
        """
        self.mapper = mapper or LayLanguageMapper()
        self._forum_complaints = self._generate_mock_forum_complaints()
    
    def _generate_mock_forum_complaints(self) -> List[str]:
        """Generate mock forum-style complaints with condition-specific symptoms."""
        templates = [
            "I've been {lay1} and {lay2} for days now",
            "Can't stop {lay1}, also have {lay2}",
            "Really worried about {lay1} and {lay2}",
            "Help! {lay1} and getting worse",
            "My {lay1} won't go away, also {lay2}",
            "Been {lay1} since yesterday, now {lay2} too",
            "Having {lay1}, plus {lay2}. Getting scared.",
            "{lay1} and {lay2}. Should I go to doctor?",
            "Anyone else with {lay1}? I also have {lay2}",
            "{lay1} is driving me crazy. {lay2} started today."
        ]
        
        # Group symptoms by related conditions for more realistic combinations
        symptom_groups = [
            # Respiratory/breathing issues
            ["can't breathe", "wheezy", "tight chest", "gasping for air", "short of breath"],
            # Cough-related
            ["coughing up stuff", "hacking cough", "dry cough", "chest hurts"],
            # Fever/infection
            ["burning up", "really tired", "aching all over", "exhausted"],
            # Upper respiratory
            ["stuffy nose", "scratchy throat", "blocked nose", "throat hurts"]
        ]
        
        complaints = []
        
        for _ in range(50):
            template = random.choice(templates)
            
            # 70% chance to pick from same symptom group (realistic combination)
            # 30% chance to mix groups (also realistic but less common)
            if random.random() < 0.7:
                symptom_group = random.choice(symptom_groups)
                if len(symptom_group) >= 2:
                    lay1, lay2 = random.sample(symptom_group, 2)
                else:
                    lay1 = random.choice(symptom_group)
                    lay2 = random.choice([t for group in symptom_groups for t in group if t != lay1])
            else:
                # Mix from different groups
                group1 = random.choice(symptom_groups)
                group2 = random.choice([g for g in symptom_groups if g != group1])
                lay1 = random.choice(group1)
                lay2 = random.choice(group2)
            
            complaint = template.format(lay1=lay1, lay2=lay2)
            
            # Fix grammar for action verbs
            if "Can't stop can't" in complaint:
                complaint = complaint.replace("Can't stop can't", "Can't")
            if "Having can't" in complaint:
                complaint = complaint.replace("Having can't", "Can't")
            
            complaints.append(complaint)
        
        return complaints
    
    def augment_complaints_with_lay_terms(
        self, 
        complaints: List[str], 
        condition_codes: List[str]
    ) -> List[str]:
        """
        Augment medical complaints with lay language.
        
        Args:
            complaints: List of medical complaints
            condition_codes: Corresponding ICD-10 codes
            
        Returns:
            List of augmented complaints
        """
        augmented = []
        
        for complaint in complaints:
            # Replace medical terms with lay terms
            augmented_complaint = complaint
            for medical, lay_list in self.mapper.reverse_mappings.items():
                if medical.replace('_', ' ') in augmented_complaint.lower():
                    lay_term = random.choice(lay_list)
                    augmented_complaint = augmented_complaint.replace(
                        medical.replace('_', ' '), 
                        lay_term
                    )
            augmented.append(augmented_complaint)
        
        return augmented
    
    def get_forum_complaints_for_pretraining(self, max_complaints: int = 100) -> List[str]:
        """
        Get forum complaints for pretraining.
        
        Args:
            max_complaints: Maximum number of complaints
            
        Returns:
            List of forum-style complaints
        """
        return self._forum_complaints[:max_complaints]


# Convenience functions
def create_forum_scraper(cache_dir: Optional[str] = None) -> ForumScraper:
    """Create a ForumScraper instance."""
    return ForumScraper(cache_dir=cache_dir)


def create_lay_language_mapper() -> LayLanguageMapper:
    """Create a LayLanguageMapper instance."""
    return LayLanguageMapper()


def create_data_augmentation(mapper: Optional[LayLanguageMapper] = None) -> ForumDataAugmentation:
    """Create a ForumDataAugmentation instance."""
    return ForumDataAugmentation(mapper=mapper)
