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
        
        lay_symptoms = [
            ("can't breathe", "dyspnea"),
            ("wheezy", "wheezing"),
            ("tight chest", "chest_tightness"),
            ("coughing up stuff", "productive_cough"),
            ("chest hurts", "chest_pain"),
            ("really tired", "fatigue"),
            ("stuffy nose", "nasal_congestion"),
            ("scratchy throat", "sore_throat")
        ]
        
        times = ["2 days", "a week", "3 hours", "yesterday", "this morning"]
        relations = ["kid", "mom", "dad", "partner", "friend"]
        
        for i in range(max_posts):
            lay_term, medical_term = random.choice(lay_symptoms)
            other_lay, other_medical = random.choice(lay_symptoms)
            
            template = random.choice(self._post_templates)
            content = template.format(
                symptom=lay_term,
                other_symptom=other_lay,
                time=random.choice(times),
                detail=random.choice(self._details),
                relation=random.choice(relations),
                diagnosis="pneumonia" if "cough" in lay_term else "asthma"
            )
            
            post = ForumPost(
                id=f"reddit_{i}",
                title=f"Question about {lay_term}",
                content=content,
                timestamp=datetime.now().isoformat(),
                forum_source="reddit",
                lay_terms=[lay_term, other_lay],
                extracted_symptoms=[medical_term, other_medical],
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
        """Generate mock forum-style complaints."""
        templates = [
            "I've been {lay1} and {lay2} for days now",
            "Can't stop {lay1}, also have {lay2}",
            "Really worried about {lay1} and {lay2}",
            "Help! {lay1} and getting worse",
            "My {lay1} won't go away, also {lay2}"
        ]
        
        lay_terms = list(self.mapper.mappings.keys())
        complaints = []
        
        for _ in range(50):
            template = random.choice(templates)
            lay1 = random.choice(lay_terms)
            lay2 = random.choice([t for t in lay_terms if t != lay1])
            complaints.append(template.format(lay1=lay1, lay2=lay2))
        
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
