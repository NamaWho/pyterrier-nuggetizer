from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class NuggetMode(Enum):
    ATOMIC = "atomic"
    NOUN_PHRASE = "noun_phrase"
    QUESTION = "question"

class NuggetScoreMode(Enum):
    VITAL_OKAY = "vital_okay"

class NuggetAssignMode(Enum):
    SUPPORT_GRADE_2 = "support_grade_2"
    SUPPORT_GRADE_3 = "support_grade_3"

@dataclass
class Query:
    """
    Represents a query.
    
    Attributes:
    - qid: Query ID
    - text: Query text
    """
    qid: str
    text: str

@dataclass
class Document:
    """
    Represents a document.
    
    Attributes:
    - docid: Document ID
    - segment: Document segment
    - title: Document title (optional)
    """
    docid: str
    segment: str
    title: Optional[str] = None

@dataclass
class Request:
    """
    Represents a request, containing a query and a list of documents.
    
    Attributes:
    - query: Query
    - documents: List of documents
    """
    query: Query
    documents: List[Document]

@dataclass
class Nugget:
    """
    Represents a nugget.

    Attributes:
    - text: Nugget text
    """
    text: str

@dataclass
class ScoredNugget(Nugget):
    """
    Represents a scored nugget.
    
    Attributes:
    - text: Nugget text
    - importance: Importance of the nugget ("vital" or "okay")
    """
    importance: str

@dataclass
class AssignedNugget(Nugget):
    """
    Represents an assigned nugget.

    Attributes:
    - text: Nugget text
    - assignment: Assignment of the nugget ("support", "not_support", or "partial_support")
    """
    assignment: str  

@dataclass
class AssignedScoredNugget(ScoredNugget):
    """
    Represents an assigned scored nugget. 
    Assignment is used to indicate the support level of the nugget, evaluting the response of the model.
    
    Attributes:
    - text: Nugget text
    - importance: Importance of the nugget ("vital" or "okay")
    - assignment: Assignment of the nugget ("support", "not_support", or "partial_support")
    """
    assignment: str  