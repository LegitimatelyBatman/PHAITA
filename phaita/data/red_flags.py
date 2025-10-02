"""Curated red-flag guidance for respiratory conditions.

Each entry maps an ICD-10 respiratory condition code to symptoms that
should trigger escalation and a short piece of clinical advice for the
triage assistant to surface to users.
"""

from typing import Dict, List, Union


RESPIRATORY_RED_FLAGS: Dict[str, Dict[str, Union[List[str], str]]] = {
    "J45.9": {
        "symptoms": [
            "inability to speak more than a few words",
            "bluish lips or fingernails",
            "peak flow < 50% of personal best"
        ],
        "escalation": "Use a rescue inhaler immediately and seek emergency care if symptoms do not improve within minutes."
    },
    "J18.9": {
        "symptoms": [
            "confusion or altered mental status",
            "rapid breathing > 30 breaths per minute",
            "oxygen saturation below 92%"
        ],
        "escalation": "Urgent evaluation in an emergency department is recommended, especially for older adults or those with chronic illness."
    },
    "J44.9": {
        "symptoms": [
            "severe breathlessness at rest",
            "worsening swelling in legs or abdomen",
            "drowsiness or new confusion"
        ],
        "escalation": "Initiate rescue medications and contact emergency services if breathing support is not available at home."
    },
    "J06.9": {
        "symptoms": [
            "stridor or noisy breathing at rest",
            "inability to swallow fluids",
            "signs of dehydration in infants or older adults"
        ],
        "escalation": "Seek urgent care if breathing becomes noisy, swallowing is impaired, or hydration cannot be maintained."
    },
    "J20.9": {
        "symptoms": [
            "high fever lasting more than 3 days",
            "bloody sputum",
            "shortness of breath at rest"
        ],
        "escalation": "Schedule prompt medical review; escalate to emergency services for breathing difficulty or blood in sputum."
    },
    "J81.0": {
        "symptoms": [
            "sudden severe breathlessness",
            "frothy pink sputum",
            "chest pain with sweating"
        ],
        "escalation": "Call emergency services immediately—acute pulmonary edema is a medical emergency."
    },
    "J93.0": {
        "symptoms": [
            "sudden stabbing chest pain",
            "rapid collapse or fainting",
            "asymmetrical chest movement"
        ],
        "escalation": "Seek emergency care without delay; a collapsed lung requires urgent intervention."
    },
    "J15.9": {
        "symptoms": [
            "persistent high fever",
            "confusion or lethargy",
            "oxygen saturation below 92%"
        ],
        "escalation": "Antibiotics may be required urgently—advise immediate clinical assessment."
    },
    "J12.9": {
        "symptoms": [
            "rapid breathing over 30 breaths per minute",
            "difficulty speaking full sentences",
            "lips or face turning blue"
        ],
        "escalation": "Advise emergency evaluation, particularly for high-risk groups such as infants and older adults."
    },
    "J21.9": {
        "symptoms": [
            "pauses in breathing (apnea)",
            "poor feeding or dehydration",
            "grunting or severe chest retractions"
        ],
        "escalation": "Infants with these signs need immediate emergency assessment and possible hospitalization."
    }
}


__all__ = ["RESPIRATORY_RED_FLAGS"]
